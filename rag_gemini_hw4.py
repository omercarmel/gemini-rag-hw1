import os
import re
import json
import faiss
import numpy as np
import nltk
from typing import List, Dict, Tuple, Optional
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
    if not os.path.isdir(folder):
        return all_sentences
    for file in os.listdir(folder):
        if file.lower().endswith(".txt"):
            path = os.path.join(folder, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
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
    top = [sentences[i] for i in I[0] if 0 <= i < len(sentences)]
    return top, D[0] if len(D) else np.array([])

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
HEADER_SCHEMA_HINT = (
    '{"covered":<bool>,"citations":[<int>...],"confidence":"high|medium|low",'
    '"suggested_queries":[<string>...],"clarifying_question":<string|null>,"reason":<string|null>}'
)

def build_prompt(enumerated_context: str, question: str, word_limit: int = 120, broad_hint: Optional[bool] = None) -> str:
    breadth_note = f"\nBREADTH_HINT: question_appears_broad={str(broad_hint).lower()}" if broad_hint is not None else ""
    return f"""
You are a retrieval-only assistant.{breadth_note}

SNIPPETS (enumerated):
{enumerated_context}

QUESTION:
{question}

INSTRUCTIONS:
1) Internally RE-RANK the snippets by relevance to the question.
2) If the answer is NOT clearly supported by the snippets, set covered=false and do not answer; provide a brief reason in "reason".
3) If supported, write a SINGLE concise answer using ONLY snippet content (no outside knowledge).
4) Add inline citations using the snippet numbers in brackets, e.g., [2][1]. Every factual claim should be supported by at least one citation.
5) Suggest 1–5 succinct better search terms for the next retrieval turn ("suggested_queries"):
   - Prefer domain nouns, key entities, synonyms, disambiguators (dates/locations).
   - Avoid stopwords and long phrases (>6 words).
6) If the user question is broad or ambiguous, propose EXACTLY ONE clarifying question in "clarifying_question"; otherwise set it to null.
7) OUTPUT STRICTLY TWO LINES with NO extra text or markdown:
   Line 1 (MINIFIED JSON header) EXACTLY with keys: {HEADER_SCHEMA_HINT}
   Line 2 (final answer OR one-line refusal if covered=false), <= {word_limit} words.
8) No URLs. No code fences. No preamble/epilogue.
""".strip()

def call_gemini(prompt: str) -> str:
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
    Expect exactly two non-empty lines:
      line 1: JSON header
      line 2: answer/refusal
    If the model returns more lines, take the first two non-empty lines.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Model output did not contain two lines.")
    return lines[0].strip(), lines[1].strip()

def parse_header_json(header_line: str) -> Dict:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", header_line, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)

# ==============================
# Validation helpers
# ==============================
def looks_broad(question: str) -> bool:
    """
    Lightweight heuristic to hint breadth to the model (non-binding).
    """
    tokens = re.findall(r"\w+", question.lower())
    if len(tokens) <= 3:
        return True
    vague = {"how", "what", "why", "info", "information", "details", "explain", "about", "tell", "guide"}
    return any(t in vague for t in tokens) and len(tokens) <= 7

def validate_header_and_answer(
    header: Dict, answer: str, available_ids: List[int]
) -> Tuple[bool, str, Dict]:
    """
    Ensures:
      - keys exist and types are correct
      - citations subset of available_ids
      - if covered=true -> answer must contain bracket citations for existing ids
      - if covered=false -> allow a one-line refusal, no invalid citations
    """
    required_keys = {"covered", "citations", "confidence", "suggested_queries", "clarifying_question", "reason"}
    if not required_keys.issubset(header.keys()):
        return False, "Header missing required keys.", header

    if not isinstance(header["covered"], bool):
        return False, "covered must be boolean.", header

    if not isinstance(header["citations"], list) or not all(isinstance(x, int) for x in header["citations"]):
        return False, "citations must be a list of ints.", header

    if header["confidence"] not in ("high", "medium", "low"):
        return False, "confidence must be one of high|medium|low.", header

    if not isinstance(header["suggested_queries"], list) or not all(isinstance(s, str) for s in header["suggested_queries"]):
        return False, "suggested_queries must be a list of strings.", header

    if header["clarifying_question"] is not None and not isinstance(header["clarifying_question"], str):
        return False, "clarifying_question must be string or null.", header

    if header["reason"] is not None and not isinstance(header["reason"], str):
        return False, "reason must be string or null.", header

    avail_set = set(available_ids)
    header["citations"] = [x for x in header["citations"] if x in avail_set]

    bracket_ids_in_answer = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    if header["covered"]:
        if not bracket_ids_in_answer:
            return False, "Covered=true but answer has no bracket citations.", header
        if not bracket_ids_in_answer.issubset(avail_set):
            return False, "Answer cites snippet ids not in available context.", header
        if not header["citations"]:
            header["citations"] = sorted(bracket_ids_in_answer)
    else:
        if bracket_ids_in_answer - avail_set:
            return False, "Refusal includes invalid citations.", header

    # Sanitize suggested_queries: unique, short
    sq = []
    seen = set()
    for s in header["suggested_queries"]:
        s2 = " ".join(s.split())
        if not s2 or len(s2.split()) > 6:
            continue
        if s2.lower() in seen:
            continue
        seen.add(s2.lower())
        sq.append(s2)
    header["suggested_queries"] = sq[:5]

    return True, "ok", header

# ==============================
# Orchestrator
# ==============================
def answer_with_citations_and_suggestions(
    question: str,
    retrieved_snippets: List[str],
) -> Tuple[Dict, str, str]:
    """
    Build enumerated context, call Gemini, parse & validate.
    Returns:
      header(dict), answer(str), enumerated_context(str)
    """
    if not retrieved_snippets:
        header = {
            "covered": False,
            "citations": [],
            "confidence": "low",
            "suggested_queries": [],
            "clarifying_question": "Can you specify the exact topic or data points you want?",
            "reason": "No relevant snippets retrieved."
        }
        refusal = "I don’t have enough evidence in the provided snippets to answer. Please provide more context or narrow the question."
        return header, refusal, ""

    enum_block, mapping = enumerate_context(retrieved_snippets, max_len_per_snippet=800)
    broad_hint = looks_broad(question)
    prompt = build_prompt(enum_block, question, word_limit=120, broad_hint=broad_hint)

    raw = call_gemini(prompt)
    try:
        header_line, answer_line = split_two_part_output(raw)
        header = parse_header_json(header_line)
    except Exception as e:
        header = {
            "covered": False,
            "citations": [],
            "confidence": "low",
            "suggested_queries": [],
            "clarifying_question": None,
            "reason": f"Model returned invalid two-part output: {e}"
        }
        answer_line = "I’m unable to produce a validated answer from the provided context."

    available_ids = [m["id"] for m in mapping]
    ok, msg, header = validate_header_and_answer(header, answer_line, available_ids)
    if not ok:
        header = {
            "covered": False,
            "citations": [],
            "confidence": "low",
            "suggested_queries": [],
            "clarifying_question": None,
            "reason": f"Validation failed: {msg}"
        }
        answer_line = "I can’t verify the answer against the provided snippets. Please refine the context or question."

    return header, answer_line, enum_block

# ==============================
# Main
# ==============================
def main():
    print("Loading documents...")
    sentences = load_documents(folder="../data")

    print("Creating FAISS index...")
    if not sentences:
        print("No documents found. Put .txt files under ../data and try again.")
        return
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = create_faiss(sentences, model)

    while True:
        q = input("\nAsk something (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        top_snips, _ = retrieve(q, model, index, sentences, k=5)
        header, answer, enum_ctx = answer_with_citations_and_suggestions(q, top_snips)

        print("\nEnumerated Context:")
        print(enum_ctx if enum_ctx else "(none)")

        print("\nHeader JSON:")
        print(json.dumps(header, ensure_ascii=False, separators=(",", ":")))

        print("\nAnswer:")
        print(answer)

        if header.get("clarifying_question"):
            print("\nClarifying question:")
            print(header["clarifying_question"])

        if header.get("suggested_queries"):
            print("\nSuggested search terms for next turn:")
            for s in header["suggested_queries"]:
                print(f"- {s}")

if __name__ == "__main__":
    main()
