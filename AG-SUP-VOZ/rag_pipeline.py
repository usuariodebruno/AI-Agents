"""
Arquivo responsável por executar o pipeline RAG (Retrieval-Augmented Generation).
Consulta o índice de documentos e chama um modelo de linguagem (LLM) 
para gerar respostas baseadas no contexto recuperado.
"""
import os
import json
import textwrap
from typing import List

from rag_query import query

# LLM client: usa OpenAI por exemplo (opcional)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT_TEMPLATE = textwrap.dedent("""
Você é um assistente que conhece o código do projeto fornecido no contexto abaixo. Use apenas o contexto para responder e não invente.
Contexto:
{context}

Pergunta:
{question}

Instruções: responda em português, seja objetivo, cite arquivos e trechos quando relevante.
""")


def assemble_context(chunks: List[dict], max_tokens_chars: int = 4000) -> str:
    out = []
    total = 0
    for c in chunks:
        t = c.get("text", "")
        if total + len(t) > max_tokens_chars:
            break
        out.append(f"Arquivo: {c.get('path')}\n{t}\n---\n")
        total += len(t)
    return "\n".join(out)


def call_llm(question: str, context_chunks: List[dict]):
    context = assemble_context(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()

    # fallback simples: concatenar contexto + pergunta
    fallback = "CONTEXT:\n" + context + "\nQUESTION:\n" + question
    return fallback


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query")
    args = p.parse_args()
    chunks = query(args.query)
    answer = call_llm(args.query, chunks)
    print(answer)
