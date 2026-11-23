# Reads the PDFs in the pdfs folder, splits them into chunks and saves the embeddings.
# Run it once, and again whenever the PDFs change.
import glob, os, json
import numpy as np
from pypdf import PdfReader
from esg_core import pick_embed_model, embed_texts

chunks = []

for path in sorted(glob.glob("pdfs/*.pdf")):
    name = os.path.basename(path)[:-4]          # file names look like samsung_2025.pdf
    company, year = name.rsplit("_", 1)
    reader = PdfReader(path)
    before = len(chunks)

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        # pieces of 1000 characters, each one starts 850 characters after the last
        for start in range(0, len(text), 850):
            piece = text[start:start + 1000]
            if len(piece) > 100:
                chunks.append({"company": company, "year": year, "page": i + 1, "text": piece})

    print(name, "->", len(chunks) - before, "chunks")

if not chunks:
    raise SystemExit("No text found. Put PDFs named like samsung_2025.pdf in the pdfs folder.")

json.dump(chunks, open("chunks.json", "w"))
print("total chunks:", len(chunks))

embed_model = pick_embed_model()
print("embedding model:", embed_model)
vectors = embed_texts([c["text"] for c in chunks], embed_model)
np.save("vectors.npy", vectors)
json.dump({"embed_model": embed_model}, open("config.json", "w"))
print("saved", vectors.shape)
