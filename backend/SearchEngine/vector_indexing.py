# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER (fixed by you)
# Organization: Centrale Méditerranée
# File        : vector_indexing.py
# Description : Indexes enriched JSONL into ChromaDB with GPU-accelerated embeddings
# Created     : 2024-05-14
# Version     : 1.4 (Fixed initialization order + robustness)
# -----------------------------------------------------------------------------

import os
import json
import time
import logging
from datetime import datetime
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"indexing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Embedding Configuration (weights used in generate_weighted_embeddings) ===
# (kept here for clarity if you want to tune)
TEXT_WEIGHT = 0.50
SUMMARY_WEIGHT = 0.35
KEYWORDS_WEIGHT = 0.15

# === Custom Fast GPU Embedder ===
class GPUEmbedder(EmbeddingFunction):
    def name(self):
        return "sentence_transformer"
    def __init__(self, model_name="multi-qa-mpnet-base-dot-v1"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"✅ Embedding model '{model_name}' loaded on: {self.device}")

    def __call__(self, texts):
        import numpy as np
        # sentence-transformers returns numpy arrays if convert_to_numpy=True
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        if embeddings is None:
            logger.warning("⚠️ Embeddings call returned None.")
            return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        if np.all(embeddings == 0):
            logger.warning("⚠️ All generated embeddings are zeros.")
        return embeddings

# === Generate Weighted Embeddings in Batch ===
def generate_weighted_embeddings(embedder, texts, summaries, keywords_list, log_samples=5):
    """
    Generate embeddings combining the full text, the summary and keywords,
    using the predefined TEXT_WEIGHT, SUMMARY_WEIGHT, KEYWORDS_WEIGHT.
    Returns a list of normalized embeddings (list-of-lists).
    """
    import numpy as np

    if not (len(texts) == len(summaries) == len(keywords_list)):
        raise ValueError("texts, summaries and keywords_list must have the same length")

    logger.info("⚙️ Encoding full texts...")
    text_embs = embedder(texts)

    logger.info("⚙️ Encoding summaries...")
    summary_embs = embedder(summaries)

    logger.info("⚙️ Encoding keywords...")
    keyword_embs = embedder(keywords_list)

    weighted_embeddings = []
    eps = 1e-10
    for idx, (t_emb, s_emb, k_emb) in enumerate(zip(text_embs, summary_embs, keyword_embs)):
        # Weighted sum
        weighted_emb = (TEXT_WEIGHT * t_emb +
                        SUMMARY_WEIGHT * s_emb +
                        KEYWORDS_WEIGHT * k_emb)

        # Normalize safely
        norm = np.linalg.norm(weighted_emb)
        if norm < eps:
            logger.warning(f"⚠️ Embedding {idx} has near-zero norm; using unnormalized vector.")
            normalized = weighted_emb.tolist()
        else:
            normalized = (weighted_emb / norm).tolist()

        weighted_embeddings.append(normalized)

        if idx < log_samples:
            logger.info(f"🔍 Embedding {idx + 1} norm: {np.linalg.norm(weighted_emb):.6f}")

    return weighted_embeddings

# === Indexer Function ===
def index_vector_store(
    jsonl_input_path: str,
    chroma_collection_name: str = "web_chunks",
    persist_directory: str = "./chroma_db",
    batch_size: int = 4000
):
    logger.info("🧠 Starting vector indexing...")
    index_start = time.time()

    embedder = GPUEmbedder()
    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=chroma_collection_name,
        embedding_function=None  # We supply embeddings manually
    )

    documents, metadatas, ids = [], [], []
    summaries, keywords_list = [], []

    try:
        with open(jsonl_input_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                summary = entry.get("summary", "")
                keywords = ", ".join(entry["keywords"]) if isinstance(entry.get("keywords", []), list) else entry.get("keywords", "")

                summaries.append(summary)
                keywords_list.append(keywords)

                documents.append(entry.get('text', ''))
                metadatas.append({
                    "title": entry.get('title', ''),
                    "url": entry.get('url', ''),
                    "keywords": keywords,
                    "summary": summary,
                    "web_path": entry.get('web_path', '')
                })
                ids.append(entry.get('id', None))

        total = len(documents)
        logger.info(f"📦 Loaded {total} documents from {jsonl_input_path}")

        # === Generate embeddings for all documents (text + summary + keywords)
        embeddings = generate_weighted_embeddings(embedder, documents, summaries, keywords_list)

        # Index in batches
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids, embeddings=batch_embs)
            logger.info(f"✅ Indexed batch {i // batch_size + 1} — {len(batch_docs)} documents")

        index_end = time.time()
        logger.info(f"✅ Vector store built in {index_end - index_start:.2f} seconds")
        logger.info(f"📚 Total documents indexed: {total}")

        if not os.path.isdir(persist_directory):
            logger.warning(f"⚠️ Warning: Expected directory '{persist_directory}' was not created.")
        else:
            logger.info(f"💾 Vector store persisted under: {persist_directory}")

    except Exception as e:
        logger.exception(f"❌ Failed to index vector store: {e}")

    logger.info("💾 Persisted vector store to disk.")

# === Entry Point ===
if __name__ == "__main__":
    import sys
    start_time = time.time()
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "enriched_pages.jsonl"
    index_vector_store(jsonl_input_path=jsonl_path)
    total = time.time() - start_time
    logger.info(f"🎉 Indexing process completed in {total:.2f} seconds")
