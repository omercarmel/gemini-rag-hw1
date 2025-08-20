import os
import faiss
import numpy as np
import nltk
import json
import re
from google import genai
from google.genai import types
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer

# -------------------------------
# Setup
# -------------------------------
# API key from env var (safer than hardcoding)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Download NLTK assets if not already installed
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Document handling
# -------------------------------
def load_documents(folder='data'):
    """Load .txt files, tokenize into sentences, clean stopwords/punctuation"""
    all_sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                for sentence in sent_tokenize(text):
                    words = word_tokenize(sentence)
                    clean = [w for w in words if w.lower() not in stop_words and w.isalnum()]
                    if clean:
                        all_sentences.append(" ".join(clean))
    return all_sentences

def create_faiss(sentences, model):
    """Embed sentences and create FAISS index"""
    embeddings = model.encode(sentences)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve(query, model, index, sentences, k=3):
    """Retrieve top-k most relevant chunks"""
    query_embed = model.encode([query])
    D, I = index.search(np.array(query_embed), k)
    return [sentences[i] for i in I[0]]

# -------------------------------
# Gemini with strict JSON output
# -------------------------------
def _extract_json(text: str) -> dict:
    """Extract first JSON object from Gemini output, strip code fences if any"""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def ask_gemini_json(context: str, question: str) -> dict:
    """
    Returns JSON with fields:
      covered: bool
      answer: str | None
      quotes: list[str]
      confidence: "high"|"medium"|"low"
      refusal_reason: str | None
    """
    if not context.strip():
        return {
            "covered": False,
            "answer": None,
            "quotes": [],
            "confidence": "low",
            "refusal_reason": "The documents do not contain information relevant to this question."
        }

    # prepare candidate short quotes
    raw_chunks = [c.strip() for c in context.split("\n") if c.strip()]
    quotes = []
    for c in raw_chunks[:3]:
        q = c.replace("\n", " ").strip()
        if len(q) > 180:
            q = q[:177] + "..."
        quotes.append(q)

    schema_json = """
{
  "covered": <boolean: true if and only if the answer is explicitly supported by the context>,
  "answer": <string or null: concise answer ONLY if covered=true; otherwise null>,
  "quotes": <array of 1 to 3 short strings directly copied from the context that support the answer; keep each <= 180 chars>,
  "confidence": <one of "high","medium","low": based on how directly and completely the context answers the question>,
  "refusal_reason": <string or null: if covered=false, a brief reason like "Missing dates in context"; otherwise null>
}
""".strip()

    prompt = f"""
You are a retrieval-only assistant.

RULES (follow exactly):
- Use ONLY the provided context. Do not add outside knowledge.
- If the context does not clearly contain the answer, set "covered": false and refuse 
  (answer=null, confidence="low", short "refusal_reason").
- If covered=true, synthesize a concise answer strictly from the context.
- Include 1–3 short quotes copied verbatim from the context that directly support the answer.
- Output ONLY VALID, MINIFIED JSON matching the schema below. No prose, no markdown, no code fences.

SCHEMA:
{schema_json}

CONTEXT:
{context}

QUESTION:
{question}

Return ONLY the JSON object.
""".strip()

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # disable hidden reasoning
        )
    )

    text = response.text.strip()
    try:
        result = _extract_json(text)
    except Exception:
        result = {
            "covered": False,
            "answer": None,
            "quotes": quotes[:1],
            "confidence": "low",
            "refusal_reason": "The model did not return valid JSON. Treat as not covered."
        }

    # Ensure defaults
    result.setdefault("covered", False)
    result.setdefault("answer", None)
    result.setdefault("quotes", [])
    result.setdefault("confidence", "low")
    result.setdefault("refusal_reason", None)

    if result.get("covered") and not result.get("quotes"):
        result["quotes"] = quotes[: min(3, len(quotes))]

    if not result.get("covered"):
        result["answer"] = None
        result["confidence"] = "low"
        if result.get("refusal_reason") is None:
            result["refusal_reason"] = "The documents do not contain information relevant to this question."

    return result

# -------------------------------
# Main loop
# -------------------------------
def main():
    print("Loading documents...")
    sentences = load_documents(folder='../data')

    print("Creating FAISS index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = create_faiss(sentences, model)

    while True:
        q = input("\nAsk something (or type 'exit'): ")
        if q.lower() == 'exit':
            break
        top_chunks = retrieve(q, model, index, sentences)
        context = "\n".join(top_chunks)
        print("\nRetrieved Context:\n", context)

        result = ask_gemini_json(context, q)
        print("\nJSON Result:")
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))

if __name__ == "__main__":
    main()