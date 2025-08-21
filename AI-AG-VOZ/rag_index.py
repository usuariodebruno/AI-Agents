"""
Arquivo para indexação de documentos usando embeddings e FAISS.
Cria um índice de similaridade para recuperação de informações.
"""

from pathlib import Path
import json
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import ast
import inspect
import re

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def read_text_files(root: str, exts=(".py", ".md", ".txt", ".rst", ".json"), exclude_dirs=("venv", ".venv", ".venv_rag", "env", ".git", "node_modules", "__pycache__")) -> List[dict]:
    docs = []
    rootp = Path(root)
    for p in rootp.rglob("*"):
        if any(part in exclude_dirs for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                docs.append({"path": str(p.relative_to(rootp)), "text": text})
            except Exception:
                continue
    return docs


def chunk_code_intelligently(file_path: str, text: str) -> List[str]:
    """
    Cria chunks de forma inteligente com base no tipo de arquivo.
    - Para Python: extrai funções e classes.
    - Para Markdown: divide por seções.
    """
    if file_path.endswith(".py"):
        chunks = []
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    try:
                        source_segment = inspect.getsource(node)
                        chunks.append(source_segment)
                    except (TypeError, OSError):
                        # Fallback para nós que não podem ter o source extraído
                        continue
            if not chunks: # Se não encontrou funções/classes, usa o texto todo
                chunks.append(text)
            return chunks
        except SyntaxError:
            return [text] # Retorna o arquivo inteiro se houver erro de sintaxe

    elif file_path.endswith(".md"):
        # Divide por títulos (##, ###, etc.)
        chunks = re.split(r'\n##+ ', text)
        return [chunk for chunk in chunks if chunk.strip()]

    else:
        # Fallback para chunking por palavras para outros tipos de arquivo
        words = text.split()
        if not words:
            return []
        
        chunk_size = 300
        overlap = 50
        chunks = []
        i = 0
        n = len(words)
        while i < n:
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks


def build_index(root_dir: str = ".", index_path: str = "index.faiss", meta_path: str = "meta.json", model_name: str = DEFAULT_MODEL, batch_size: int = 32):
    """Roda a indexação: lê arquivos, cria chunks inteligentes, gera embeddings e grava."""
    docs = read_text_files(root_dir)
    if not docs:
        print("Nenhum documento encontrado para indexar.")
        return False

    model = SentenceTransformer(model_name)

    metas = []
    all_chunks_text = []
    print("Criando chunks inteligentes dos arquivos...")
    for d in docs:
        chunks = chunk_code_intelligently(d["path"], d["text"])
        for c in chunks:
            if c.strip():
                metas.append({"path": d["path"], "text": c})
                all_chunks_text.append(c)

    if not all_chunks_text:
        print("Nenhum chunk gerado.")
        return False

    print(f"Gerando embeddings para {len(all_chunks_text)} chunks (batch_size={batch_size}) usando '{model_name}' ...")
    embeddings = model.encode(all_chunks_text, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)

    X = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(X)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    print(f"Index criado: {index_path} (dim={dim}, items={len(metas)})")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=".", help="pasta do projeto a indexar")
    p.add_argument("--index", default="index.faiss")
    p.add_argument("--meta", default="meta.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size", help="batch size para geração de embeddings")
    args = p.parse_args()
    build_index(
        args.root,
        index_path=args.index,
        meta_path=args.meta,
        model_name=args.model,
        batch_size=args.batch_size,
    )
