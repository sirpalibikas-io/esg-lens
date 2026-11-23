# Quick question from the terminal: python ask.py "your question"
import sys
from esg_core import *

chunks, vectors, embed_model = load_index()
question = " ".join(sys.argv[1:]) or "What is the net-zero target year?"
ans, hits = answer(question, chunks, vectors, embed_model)
print(ans)
print("\nSources:")
for h in hits:
    print(f"  {pretty(h['company'])} {h['year']}, page {h['page']}  (score {h['score']:.2f})")
