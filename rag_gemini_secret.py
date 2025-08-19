import os
import faiss
import numpy as np
import nltk
from google import genai
from google.genai import types
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer

# Set your Gemini API key here
client = genai.Client(api_key="api_key")

# Download NLTK assetsexit
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_documents(folder='rag with secret'):
    all_sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                for sentence in sent_tokenize(text):
                    words = word_tokenize(sentence)
                    clean = [w for w in words if w.lower() not in stop_words and w.isalnum()]
                    all_sentences.append(" ".join(clean))
    return all_sentences


def create_faiss(sentences, model):
    embeddings = model.encode(sentences)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings


def retrieve(query, model, index, sentences, k=3):
    query_embed = model.encode([query])
    D, I = index.search(np.array(query_embed), k)
    return [sentences[i] for i in I[0]]


def ask_gemini(context, question):
    prompt = f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {question}
Answer:"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable "thinking"
        )
    )
    return response.text.strip()

def main():
    print("Loading documents...")
    sentences = load_documents(folder='../rag with secret')

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
        answer = ask_gemini(context, q)
        print("\nGemini Answer:\n", answer)

if __name__ == "__main__":
    main()
