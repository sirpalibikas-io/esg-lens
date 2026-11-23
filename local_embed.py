from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)