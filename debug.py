import json
from esg_core import *

chunks, vectors, embed_model = load_index()

# 1. What did the search actually find?
question = "What is Samsung's net-zero target year?"
hits = retrieve(question, chunks, vectors, embed_model, k=5, company="samsung")
print("=== TOP 5 CHUNKS FOUND BY SEARCH ===")
for h in hits:
    print(f"\nPage {h['page']} (score {h['score']:.2f})")
    print(h["text"][:400])
    print("-----")

# 2. Does the report mention these words anywhere?
print("\n=== KEYWORD CHECK IN SAMSUNG REPORT ===")
words = ["net zero", "net-zero", "carbon neutral", "2050", "RE100"]
for w in words:
    pages = sorted({c["page"] for c in chunks
                    if c["company"] == "samsung" and w.lower() in c["text"].lower()})
    print(f"{w!r}: found on pages {pages[:15]}")