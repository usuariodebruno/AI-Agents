""""
Arquivo responsável por consultar o índice de documentos
e retornar os metadados dos documentos mais relevantes
baseados em uma consulta de texto.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_index(index_path: str = "data/index.faiss", meta_path: str = "data/meta.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    model = SentenceTransformer(DEFAULT_MODEL)
    return index, metas, model


def query(query_text: str, index_path: str = "data/index.faiss", meta_path: str = "data/meta.json", k: int = 5):
    index, metas, model = load_index(index_path, meta_path)
    q_emb = model.encode(query_text)
    q = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    results = []
    for idx in I[0]:
        if idx < len(metas):
            results.append(metas[idx])
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query")
    args = p.parse_args()
    res = query(args.query)
    for r in res:
        print(r["path"])
        print(r["text"][:400])
        print("---")
