import os
import re
import json
import faiss
import numpy as np
import nltk
from typing import List, Dict, Tuple
from google import genai
from google.genai import types
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer

# ==============================
# Setup
# ==============================
# Use env var for safety: set GEMINI_API_KEY in your environment
client = genai.Client(api_key=os.getenv("api_key"))

# NLTK assets (no-op if already present)
nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# ==============================
# Document handling
# ==============================
def load_documents(folder: str = "data") -> List[str]:
    """
    Load .txt docs from folder, split to sentences, lightly clean (remove stopwords & non-alnum tokens).
    Returns a list of cleaned sentences/chunks.
    """
    all_sentences: List[str] = []
    for file in os.listdir(folder):
        if file.lower().endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
                for sentence in sent_tokenize(text):
                    tokens = word_tokenize(sentence)
                    clean = [w for w in tokens if w.lower() not in STOP_WORDS and w.isalnum()]
                    if clean:
                        all_sentences.append(" ".join(clean))
    return all_sentences

# ==============================
# Retrieval (FAISS)
# ==============================
def create_faiss(sentences: List[str], model: SentenceTransformer) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    embeds = model.encode(sentences)
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeds))
    return index, embeds

def retrieve(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, sentences: List[str], k: int = 5) -> Tuple[List[str], np.ndarray]:
    """
    Return top-k sentences and distances (smaller is better for L2).
    """
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    top = [sentences[i] for i in I[0]]
    return top, D[0]

# ==============================
# Context preparation
# ==============================
def enumerate_context(snippets: List[str], max_len_per_snippet: int = 800) -> Tuple[str, List[Dict]]:
    """
    Turn a list of snippets into an enumerated block:
        [1] snippet1
        [2] snippet2
    Also returns structured mapping [{"id":1,"text":...}, ...]
    """
    mapping = []
    lines = []
    for idx, s in enumerate(snippets, start=1):
        s_norm = " ".join(s.split())
        if len(s_norm) > max_len_per_snippet:
            s_norm = s_norm[: max_len_per_snippet - 3] + "..."
        mapping.append({"id": idx, "text": s_norm})
        lines.append(f"[{idx}] {s_norm}")
    return "\n".join(lines), mapping

# ==============================
# Gemini prompt + parsing
# ==============================
HEADER_SCHEMA_HINT = """
{"covered":<bool>,"use_snippets":[<int>...],"citations":[<int>...],"confidence":"high|medium|low","reason":<string|null>}
""".strip()

def build_prompt(enumerated_context: str, question: str, word_limit: int = 120) -> str:
    return f"""
You are a retrieval-only assistant.

SNIPPETS (enumerated):
{enumerated_context}

QUESTION:
{question}

INSTRUCTIONS:
1) Internally RE-RANK the snippets by relevance to the question.
2) If the answer is NOT clearly supported by the snippets, set covered=false and do not answer; provide a brief reason in "reason".
3) If supported, write a SINGLE concise answer using ONLY snippet content (no outside knowledge).
4) Add inline citations using the snippet numbers in brackets, e.g., [2][1]. Every factual claim should be supported by at least one citation.
5) OUTPUT STRICTLY in TWO parts with NO extra text or markdown:
   Part A (FIRST line ONLY) — a MINIFIED JSON header EXACTLY matching this shape and keys:
   {HEADER_SCHEMA_HINT}
   Part B (SECOND line ONLY) — the final answer text OR a one-line polite refusal if covered=false.
6) Keep the answer under {word_limit} words.

Rules:
- No URLs. No markdown fences. No preamble or epilogue. Only the two lines described above.
- Citations must reference only the provided enumerated snippet numbers.
""".strip()

def call_gemini_two_part(prompt: str) -> str:
    resp = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    return resp.text.strip()

def split_two_part_output(text: str) -> Tuple[str, str]:
    """
    Expect exactly two lines:
      line 1: JSON header
      line 2: answer/refusal
    If the model returns more lines, take the first two non-empty lines.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Model output did not contain two lines.")
    header = lines[0].strip()
    answer = lines[1].strip()
    return header, answer

def parse_header_json(header_line: str) -> Dict:
    """
    Remove any accidental code-fence markers and parse JSON.
    """
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", header_line, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)

# ==============================
# Validation helpers
# ==============================
def validate_header_and_answer(
    header: Dict, answer: str, available_ids: List[int]
) -> Tuple[bool, str, Dict]:
    """
    Ensures:
      - keys exist and types are correct
      - citations subset of available_ids
      - if covered=false -> answer should be refusal; if covered=true -> answer should have bracket citations that map to available ids
    Returns (ok, message, possibly_fixed_header)
    """
    required_keys = {"covered", "use_snippets", "citations", "confidence", "reason"}
    if not required_keys.issubset(header.keys()):
        return False, "Header missing required keys.", header

    if not isinstance(header["covered"], bool):
        return False, "covered must be boolean.", header

    if not isinstance(header["use_snippets"], list) or not all(isinstance(x, int) for x in header["use_snippets"]):
        return False, "use_snippets must be a list of ints.", header

    if not isinstance(header["citations"], list) or not all(isinstance(x, int) for x in header["citations"]):
        return False, "citations must be a list of ints.", header

    if header["confidence"] not in ("high", "medium", "low"):
        return False, "confidence must be one of high|medium|low.", header

    if header["reason"] is not None and not isinstance(header["reason"], str):
        return False, "reason must be string or null.", header

    # Clamp invalid ids to available ids
    avail_set = set(available_ids)
    header["use_snippets"] = [x for x in header["use_snippets"] if x in avail_set]
    header["citations"] = [x for x in header["citations"] if x in avail_set]

    # Basic answer checks
    bracket_ids_in_answer = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    if header["covered"]:
        if not bracket_ids_in_answer:
            return False, "Covered=true but answer has no bracket citations.", header
        if not bracket_ids_in_answer.issubset(avail_set):
            return False, "Answer cites snippet ids not in available context.", header
        if not header["citations"]:
            # If model forgot to fill 'citations', infer from answer
            header["citations"] = sorted(bracket_ids_in_answer)
    else:
        # refusal path: allow any short refusal; still prefer no invalid citations
        if bracket_ids_in_answer - avail_set:
            return False, "Refusal includes invalid citations.", header

    return True, "ok", header

# ==============================
# Orchestrator
# ==============================
def answer_with_citations(
    question: str,
    retrieved_snippets: List[str],
    distances: np.ndarray
) -> Tuple[bool, Dict, str, str]:
    """
    Build enumerated context, call Gemini, parse & validate.
    Returns:
      success(bool), header(dict), answer(str), enumerated_context(str)
    """
    if not retrieved_snippets:
        # No evidence -> refusal without calling the model
        header = {
            "covered": False,
            "use_snippets": [],
            "citations": [],
            "confidence": "low",
            "reason": "No relevant snippets retrieved."
        }
        refusal = "I can’t find enough evidence in the provided snippets to answer that. Please add more relevant context."
        return True, header, refusal, ""

    enum_block, mapping = enumerate_context(retrieved_snippets, max_len_per_snippet=800)
    prompt = build_prompt(enum_block, question, word_limit=120)

    raw = call_gemini_two_part(prompt)
    try:
        header_line, answer_line = split_two_part_output(raw)
        header = parse_header_json(header_line)
    except Exception as e:
        # Hard fallback if the model violates the two-part protocol
        header = {
            "covered": False,
            "use_snippets": [],
            "citations": [],
            "confidence": "low",
            "reason": f"Model returned invalid two-part output: {e}"
        }
        answer_line = "I’m unable to produce a validated answer from the provided context."

    available_ids = [m["id"] for m in mapping]
    ok, msg, header = validate_header_and_answer(header, answer_line, available_ids)
    if not ok:
        # Final safe fallback
        header = {
            "covered": False,
            "use_snippets": [],
            "citations": [],
            "confidence": "low",
            "reason": f"Validation failed: {msg}"
        }
        answer_line = "I can’t verify the answer against the provided snippets. Please refine the context or question."

    return True, header, answer_line, enum_block

# ==============================
# Main
# ==============================
def main():
    print("Loading documents...")
    sentences = load_documents(folder="../data")

    print("Creating FAISS index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = create_faiss(sentences, model)

    while True:
        q = input("\nAsk something (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        top_snips, dists = retrieve(q, model, index, sentences, k=5)
        success, header, answer, enum_ctx = answer_with_citations(q, top_snips, dists)

        print("\nEnumerated Context:")
        print(enum_ctx if enum_ctx else "(none)")

        print("\nHeader JSON:")
        print(json.dumps(header, ensure_ascii=False, separators=(",", ":")))

        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()
