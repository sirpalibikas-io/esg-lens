import os, re, time, json
import numpy as np
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastembed import TextEmbedding
from dotenv import load_dotenv

# reads GEMINI_API_KEY from .env on your Mac, or from Streamlit Secrets in the cloud
load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY not found. Add it to the .env file (see .env.example).")

# Gemini writes the final answer; if one model is busy, the next one is tried
GEN_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

TEMPORARY = ["429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "404", "NOT_FOUND"]

# same embedding model as build_index.py (all-MiniLM-L6-v2), but light enough for the cloud
_embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


def pretty(company):
    return company.replace("_", " ").title()


def embed_texts(texts, model=None):
    vecs = np.array(list(_embedder.embed(list(texts))), dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def generate(prompt, system=None, rounds=3):
    last = None
    for r in range(rounds):
        for m in GEN_MODELS:
            for attempt in range(2):
                try:
                    res = client.models.generate_content(
                        model=m, contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=system, temperature=0))
                    return (res.text or "").strip()
                except Exception as e:
                    last = e
                    if not any(s in str(e) for s in TEMPORARY):
                        raise
                    if "404" in str(e) or "NOT_FOUND" in str(e):
                        break               # model name not available, go to the next model
                    print(f"{m} busy, retrying...")
                    time.sleep(3)
        time.sleep(10)                      # all models busy, wait a bit and go around again
    raise last


def load_index():
    chunks = json.load(open("chunks.json"))
    vectors = np.load("vectors.npy")
    embed_model = json.load(open("config.json"))["embed_model"]
    return chunks, vectors, embed_model


def detect_company(question, chunks):
    q = question.lower()
    names = {c["company"] for c in chunks}
    found = [n for n in names if n.replace("_", " ") in q]
    return found[0] if len(found) == 1 else None


# keyword search (TF-IDF) is built once and reused
_tfidf = {}

def get_tfidf(chunks):
    if "vec" not in _tfidf:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        _tfidf["X"] = vec.fit_transform([c["text"] for c in chunks])
        _tfidf["vec"] = vec
    return _tfidf["vec"], _tfidf["X"]


def retrieve(question, chunks, vectors, embed_model, k=8, company=None):
    idx = [i for i, c in enumerate(chunks) if company is None or c["company"] == company]

    # 1. meaning search (embeddings + cosine similarity)
    q = embed_texts([question], embed_model)
    dense = cosine_similarity(q, vectors[idx])[0]

    # 2. keyword search (TF-IDF), catches exact words like "net zero"
    vec, X = get_tfidf(chunks)
    sparse = cosine_similarity(vec.transform([question]), X[idx])[0]

    # 3. merge the two rankings (reciprocal rank fusion)
    dense_rank = np.argsort(np.argsort(-dense))
    sparse_rank = np.argsort(np.argsort(-sparse))
    fused = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)

    top = np.argsort(-fused)[:k]
    return [dict(chunks[idx[t]], score=float(dense[t])) for t in top]


SYSTEM = (
    "You answer questions about corporate sustainability reports. "
    "Use only the context you are given and never add facts from memory. "
    "Write a short answer and put the source after each fact, like (Samsung 2025, p. 42). "
    "If the context does not contain the answer, reply exactly: Not disclosed in the report."
)


def answer(question, chunks, vectors, embed_model, company=None, k=8):
    company = company or detect_company(question, chunks)
    hits = retrieve(question, chunks, vectors, embed_model, k=k, company=company)
    context = "\n\n".join(
        f"[{pretty(h['company'])} {h['year']}, page {h['page']}]\n{h['text']}" for h in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return generate(prompt, SYSTEM), hits