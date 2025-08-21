"""
Arquivo responsável por carregar o modelo de IA, 
o tokenizer e as respostas.
"""

import os
import json
import re
import numpy as np
import sys

from data.qa_data import qa_pairs
from core.utils import criar_tokenizer, texto_para_sequencia
from services.rag_query import query as rag_query

# Adicionando o diretório raiz do projeto ao sys.path para corrigir problemas de importação relativa após a modularização.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuração RAG e LLM ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
RAG_ENABLED = os.path.exists("data/index.faiss") and os.path.exists("data/meta.json")
LLM_AVAILABLE = False

# Configura o provedor de LLM selecionado
try:
    if LLM_PROVIDER == "openai":
        import openai
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        if OPENAI_API_KEY:
            LLM_AVAILABLE = True
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

PROMPT_TEMPLATE = """Você é um assistente especialista em traduzir funcionalidades de software para um público **leigo e não técnico**. Sua missão é usar o contexto de código de um sistema (front-end e back-end) para explicar **o que o sistema faz e como usá-lo, do ponto de vista de um usuário final**.

**Instruções Fundamentais:**
1.  **Traduza o Código em Ações do Usuário:** Analise o contexto de código para entender a funcionalidade. Descreva as ações que um usuário pode realizar na interface e os benefícios que ele obtém, sem mencionar a lógica de programação por trás.
2.  **Foque no "O quê" e "Para quê":** Explique o que a funcionalidade permite ao usuário fazer e por que isso é útil para ele.
3.  **Linguagem Simples e Direta:** Use uma linguagem acessível. Se a pergunta for sobre criar, editar ou excluir algo, forneça um guia prático e direto.
4.  **Crie um Passo a Passo Preciso (Baseado no Código):** Para guias de usuário, utilize o contexto de código para fornecer um caminho exato.
    -   Identifique no código o nome exato dos menus, botões, seções e campos da tela.
    -   Descreva a sequência de cliques e ações que o usuário deve realizar. Não use termos genéricos como "clique no menu principal" ou "o nome do botão pode variar". Seja específico.
5.  **Gere Sugestões Relevantes:** Ao final da explicação, crie 2 ou 3 sugestões de perguntas que o usuário poderia ter sobre funcionalidades relacionadas, incentivando a exploração do sistema.

**Restrições:**
-   Proibido usar jargão técnico (ex: API, endpoint, função, variável, classe, componente).
-   Proibido explicar a implementação ou "como o código funciona". Foque apenas no resultado visível para o usuário.
-   Proibido usar os termos "código", "programação" ou "desenvolvimento".
-   Proibido usar asteriscos (*) ou qualquer outro marcador que não sejam números para listas.
-   Proibido usar frases vagas ou suposições. A resposta deve ser baseada estritamente no contexto fornecido.

---

**Exemplo de Saída Ideal:**

Para criar um novo evento na plataforma, siga este mapa de navegação e passo a passo:

**Mapa do Site:** Tela Inicial > Menu "Eventos" > Botão "Criar Novo Evento" > Página "Formulário de Criação"

**Passo a Passo:**
Etapa 1.  Após fazer seu login, acesse o menu lateral esquerdo e clique na opção **"Eventos"**.
Etapa 2.  Na tela de listagem de eventos, procure e clique no botão azul chamado **"Criar Novo Evento"**, localizado no canto superior direito.
Etapa 3.  Você será direcionado para a página **"Formulário de Criação"**. Preencha os seguintes campos:
    -   **Título do Evento:** Dê um nome claro e atrativo.
    -   **Detalhes Completos:** Descreva tudo sobre seu evento para os participantes.
    -   **Data e Horário:** Especifique o dia e a hora de início e término.
    -   **Localização:** Informe o endereço físico ou o link de acesso, caso seja online.
    -   **Capa do Evento (Opcional):** Clique em "Escolher Arquivo" para adicionar uma imagem representativa.
Etapa 4.  Após preencher tudo, revise as informações e clique no botão verde **"Publicar Evento"** no final da página para que ele fique visível a todos.

**SUGESTÕES:**
-   Como faço para editar as informações de um evento que já publiquei?
-   É possível ver uma lista de todas as pessoas que se inscreveram no meu evento?
-   O que acontece se eu precisar cancelar um evento?

---

**Contexto:**
{context}

**Pergunta:**
{question}

"""
# --- Fim da Configuração RAG ---

# Tentar carregar artefatos salvos
MODEL_PATH = "models/model.h5"
TOKENIZER_PATH = "models/tokenizer.json"
RESPOSTAS_PATH = "data/respostas.json"

if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(RESPOSTAS_PATH):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(RESPOSTAS_PATH, "r", encoding="utf-8") as f:
        respostas = json.load(f)

    # preparar perguntas a partir do qa_pairs para mapa (mas respostas vem do arquivo)
    perguntas = list(qa_pairs.keys())
else:
    # fallback: treinar rapidamente em tempo de import (quando artefatos não existirem)
    import tensorflow as tf
    tokenizer = criar_tokenizer(qa_pairs)
    perguntas = list(qa_pairs.keys())
    respostas = list(qa_pairs.values())
    X = np.array([texto_para_sequencia(tokenizer, p)[0] for p in perguntas])
    y = np.arange(len(respostas))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=8, input_length=10),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(respostas), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=200, verbose=0)

    respostas = list(qa_pairs.values())

# Criar mapa de perguntas normalizadas para correspondência rápida
def _normalize(texto: str):
    texto = texto.lower().strip()
    texto = re.sub(r"[^0-9a-zãáâàéêíóôõúüç\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto

_norm_qa_map = { _normalize(k): v for k, v in qa_pairs.items() }


def _summarize_chunks_fallback(chunks: list) -> str:
    """
    Cria um resumo estruturado e mais natural a partir dos chunks encontrados,
    servindo como um fallback inteligente quando o LLM não está disponível.
    """
    summary_parts = []
    techs = set()
    files_mentioned = set()

    # Tenta extrair tecnologias de arquivos de dependência
    for chunk in chunks:
        filename = os.path.basename(chunk['path']).lower()
        if 'requirements.txt' in filename:
            # Extrai pacotes principais do requirements.txt
            lines = chunk['text'].split('\n')
            for line in lines:
                if 'django' in line.lower(): techs.add('Django')
                if 'react' in line.lower(): techs.add('React')
                if 'flask' in line.lower(): techs.add('Flask')
        files_mentioned.add(chunk['path'])

    # Usa o primeiro chunk (geralmente o mais relevante) para uma descrição
    if chunks:
        first_chunk_text = chunks[0]['text']
        # Pega as primeiras frases como uma descrição geral
        description = ". ".join(first_chunk_text.split('.')[:3])
        summary_parts.append(f"Analisando o arquivo '{chunks[0]['path']}', parece que o projeto é sobre o seguinte: \"{description.strip()}...\"")

    if techs:
        summary_parts.append(f"Ele parece utilizar tecnologias como: {', '.join(sorted(list(techs)))}.")

    summary_parts.append(f"Encontrei essas informações nos arquivos: {', '.join(files_mentioned)}.")

    final_summary = "\n".join(summary_parts)
    
    return "Não consegui consultar o modelo de linguagem, mas com base nos arquivos, posso te adiantar o seguinte:\n\n" + final_summary


def responder_com_rag(pergunta: str, k: int = 3):
    """Busca no índice vetorial e retorna uma tupla (resposta, sugestões)."""
    print("\n[INFO] Buscando na base de código (RAG)...")
    chunks = rag_query(pergunta, k=k)
    
    if not chunks:
        print("[INFO] Nenhum contexto relevante encontrado no RAG.")
        return None

    contexto = "\n\n".join(c["text"] for c in chunks)

    if not LLM_AVAILABLE:
        print(f"[INFO] Provedor de LLM '{LLM_PROVIDER}' não configurado. Retornando contexto encontrado.")
        # Retorna o contexto encontrado como fallback
        fallback_response = "O modelo de linguagem não está configurado, mas encontrei as seguintes informações relevantes no código:\n\n"
        for i, chunk in enumerate(chunks[:2]):
            fallback_response += f"--- Trecho {i+1} do arquivo '{chunk['path']}' ---\n{chunk['text']}\n\n"
        return fallback_response.strip()

    print(f"[INFO] Contexto encontrado. Consultando LLM via '{LLM_PROVIDER}'...")
    prompt = PROMPT_TEMPLATE.format(context=contexto, question=pergunta)

    try:
        if LLM_PROVIDER == "openai":
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            raw_response = resp.choices[0].message.content.strip()
            return _parse_llm_response(raw_response)

        elif LLM_PROVIDER == "gemini":
            # Reordenar para priorizar modelos com limites mais altos na camada gratuita
            gemini_models = ['gemini-1.5-flash', 'gemini-1.0-pro', 'gemini-1.5-pro']
            
            last_error = None
            for model_name in gemini_models:
                try:
                    print(f"[INFO] Tentando modelo: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    print(f"[INFO] Sucesso com o modelo: {model_name}")
                    raw_response = response.text.strip()
                    return _parse_llm_response(raw_response)
                except Exception as e:
                    print(f"[AVISO] Falha no modelo {model_name}: {e}")
                    last_error = e
                    continue  # Tenta o próximo modelo da lista
            
            # Se todos os modelos falharam, levanta a última exceção
            raise last_error

    except Exception as e:
        print(f"[ERRO] Falha ao chamar a API do '{LLM_PROVIDER}': {e}")
        fallback_answer = _summarize_chunks_fallback(chunks)
        return fallback_answer, []  # Retorna tupla com lista de sugestões vazia

def responder(texto_usuario):
    """Função principal para obter uma tupla (resposta, sugestões)."""
    texto_norm = _normalize(texto_usuario)

    # 1. Correspondência direta
    if texto_norm in _norm_qa_map:
        print("\n[INFO] Fonte da resposta: Correspondência Direta (qa_data).")
        return _norm_qa_map[texto_norm], []

    # 2. Modelo de ML
    print("\n[INFO] Fonte da resposta: Modelo de ML.")
    seq = texto_para_sequencia(tokenizer, texto_usuario)
    pred = model.predict(seq, verbose=0)
    idx = np.argmax(pred)
    prob = np.max(pred)

    CONFIDENCE_THRESHOLD = 0.75  # Limite de confiança

    if prob > CONFIDENCE_THRESHOLD:
        print(f"[INFO] Confiança do modelo: {prob:.2f} (acima do limite de {CONFIDENCE_THRESHOLD})")
        return respostas[idx], []
    
    print(f"[INFO] Confiança do modelo: {prob:.2f} (abaixo do limite de {CONFIDENCE_THRESHOLD})")

    # 3. Fallback para RAG se o índice existir
    if RAG_ENABLED:
        rag_result = responder_com_rag(texto_usuario)
        if rag_result:
            return rag_result  # Propaga a tupla (resposta, sugestões)

    # 4. Resposta final de fallback
    print("[INFO] Fonte da resposta: Fallback Padrão.")
    return "Desculpe, não tenho certeza de como responder a isso. Pode reformular a pergunta?", []

def _parse_llm_response(response: str) -> tuple[str, list[str]]:
    """
    Analisa a resposta do LLM para separar a resposta principal das sugestões.
    Retorna uma tupla contendo (resposta_principal, lista_de_sugestoes).
    """
    sugestoes = []
    # A regex busca por "SUGESTÕES:" (case-insensitive) e captura tudo depois
    match = re.search(r"SUGESTÕES:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
    
    if match:
        resposta_principal = response[:match.start()].strip()
        sugestoes_texto = match.group(1).strip()
        # Divide as sugestões por nova linha e remove itens vazios ou marcadores
        sugestoes = [s.strip().lstrip('-* ').strip() for s in sugestoes_texto.split('\n') if s.strip()]
    else:
        resposta_principal = response.strip()

    return resposta_principal, sugestoes
