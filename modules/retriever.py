import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from modules.schemas import DocumentChunk

class DocumentRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required.")

        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Build the retrieval index before searching.")
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks)),
        )

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            chunk = self.chunks[int(index)]
            results.append({
                "chunk": chunk,
                "score": float(score),
            })

        return results
