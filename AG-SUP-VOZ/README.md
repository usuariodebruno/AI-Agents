# Guia do Desenvolvedor: Agente de IA Conversacional

Este documento detalha a arquitetura, componentes e fluxos de trabalho do projeto, servindo como um guia para manutenção e desenvolvimento.

## Visão Geral

O projeto é um agente de IA conversacional modular e extensível, projetado para interagir com usuários através de texto e voz. Suas principais funcionalidades incluem:

1.  **Gerenciamento de Conhecimento**: Responde a perguntas utilizando uma base de conhecimento flexível, que pode ser carregada de um cache local, dados embutidos ou dinamicamente de uma **API externa** (configurável via `qa_data.py`).
2.  **Processamento de Linguagem Natural (PLN)**: Emprega um modelo de Machine Learning local para classificação de intenção, permitindo respostas inteligentes mesmo para perguntas não exatas.
3.  **Síntese de Voz (TTS)**: Converte respostas textuais em áudio de forma assíncrona, utilizando bibliotecas como `edge-tts` ou `pyttsx3`, com um sistema de cache (`tts_cache/`) para garantir baixa latência e eficiência na reprodução.
4.  **Geração Aumentada por Recuperação (RAG)**: Opcionalmente, pode analisar e responder a perguntas complexas sobre bases de código (incluindo seu próprio código-fonte ou projetos externos), integrando-se a Large Language Models (LLMs) como OpenAI ou Google Gemini para gerar respostas contextuais.

---

## Arquitetura e Fluxo de Dados

O agente opera com uma arquitetura em camadas que prioriza velocidade e precisão, com fallbacks inteligentes.

#### Diagrama de Arquitetura

```
      +------------------+
      |   Usuário (CLI)  |
      +--------+---------+
               |
               v
      +--------+---------+
      |    main.py       | (Loop Principal de Interação)
      +--------+---------+
               |
               v
      +--------+---------+
      |   model.py       | (Função `responder`)
      +--------+---------+
               |
     +----------------------------------------------------------------+
     |         Lógica de Resposta (em ordem de execução)              |
     |                                                                |
     |  1. Busca Rápida: Correspondência direta em `qa_pairs`?         |
     |     |                                                          |
     |     +-- Sim -> Resposta encontrada.                            |
     |     |                                                          |
     |     +-- Não -> Passo 2.                                        |
     |                                                                |
     |  2. Modelo ML: Usar `model.predict()` para classificar intenção?|
     |     |                                                          |
     |     +-- Confiança > Limite -> Resposta encontrada.             |
     |     |                                                          |
     |     +-- Confiança < Limite -> Passo 3 (Opcional).              |
     |                                                                |
     |  3. Pipeline RAG: A pergunta é sobre o código?                 |
     |     |                                                          |
     |     +--> rag_query.py (Busca no índice FAISS)                   |
     |     |      |                                                   |
     |     |      v                                                   |
     |     +--> rag_pipeline.py (Monta prompt + chama LLM)             |
     |            |                                                   |
     |            +--> Resposta gerada pelo LLM.                      |
     |                                                                |
     +----------------------------------------------------------------+
               |
               v
      +--------+---------+
      |    main.py       | (Recebe texto da resposta)
      +--------+---------+
               |
               v
      +--------+---------+
      |   Fila de Fala   | (Queue)
      +--------+---------+
               |
               v
      +------------------+     +------------------+
      |  Worker de TTS   |---->|  Cache de Áudio  |
      | (edge-tts/pyttsx3)|     |   (tts_cache/)   |
      +------------------+     +------------------+
               |
               v
      +--------+---------+
      | Player de Áudio  | (mpg123, ffplay, etc.)
      +------------------+

```

---

## Ambientes Virtuais e Modos de Uso

O projeto é projetado para ser flexível, oferecendo dois modos de operação com ambientes virtuais separados para evitar conflitos de dependência.

#### Modo 1: Agente Local (Sem LLM)
-   **Funcionalidade:** O agente responde apenas com base na sua base de conhecimento pré-definida (`qa_data.py`) e no modelo de classificação local.
-   **Vantagens:** Rápido, funciona totalmente offline, sem custos de API.
-   **Ambiente Necessário:** Apenas o `.venv`.

#### Modo 2: Agente com RAG + LLM
-   **Funcionalidade:** Adiciona a capacidade de analisar uma base de código, buscar informações relevantes e gerar respostas com um LLM (OpenAI ou Gemini).
-   **Vantagens:** Capaz de responder perguntas complexas sobre código que não estão na base de conhecimento.
-   **Ambientes Necessários:** **Ambos**, `.venv` para o agente e `.venv_rag` para a pipeline de análise.

---

## Primeiros Passos (Instalação)

1.  **Clone o repositório.**

2.  **Instale as dependências do sistema** (se necessário):
    ```bash
    sudo apt update
    sudo apt install python3-venv python3-dev build-essential mpg123
    ```

3.  **Configure o ambiente principal (`.venv`) - Essencial para todos os usos:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Configure o ambiente RAG (`.venv_rag`) - Opcional, para análise de código:**
    ```bash
    python3 -m venv .venv_rag
    source .venv_rag/bin/activate
    pip install -r requirements-rag.txt
    ```

---

## Fluxos de Trabalho Essenciais

#### 1. Treinar o Modelo de Classificação

Para evitar o treinamento lento durante a inicialização, gere os artefatos do modelo offline.

```bash
# Ative o ambiente principal
source .venv/bin/activate

# Execute o script de treino
python training/train.py
```
Isso criará `models/model.h5`, `models/tokenizer.json` e `data/respostas.json`.

#### 2. Indexar a Base de Código (RAG)

Para que o agente possa responder perguntas sobre uma base de código, você precisa primeiro criar um índice vetorial a partir dos arquivos-fonte. Este passo é crucial e define **qual projeto** o agente irá analisar.

```bash
# Ative o ambiente RAG
source .venv_rag/bin/activate
```

**Opção A: Indexar o próprio código do Agente (Padrão)**

Use este comando para fazer o agente analisar seu próprio código-fonte. O `.` no comando significa "o diretório atual".

```bash
# Rode o indexador no diretório atual
python training/rag_index.py .
```

**Opção B: Indexar um Projeto Externo**

Para fazer o agente analisar **outro sistema**, substitua `.` pelo caminho completo da pasta desse projeto.

```bash
# Exemplo: Indexando um projeto localizado em /home/usuario/projetos/meu-sistema
python training/rag_index.py /home/usuario/projetos/meu-sistema
```

Isso irá ler os arquivos do outro projeto e **sobrescrever** os arquivos `data/index.faiss` e `data/meta.json` com a nova base de conhecimento. Execute este comando sempre que o código-fonte que você quer analisar for alterado significativamente.

#### 3. Executar o Agente

Com os artefatos prontos, inicie a aplicação principal.

```bash
# Ative o ambiente principal
source .venv/bin/activate

# Inicie o agente
python core/main.py
```

---

## Detalhamento dos Componentes

-   `core/main.py`: Ponto de entrada. Gerencia o loop de interação com o usuário, chama o modelo para obter respostas e coordena a síntese e reprodução de áudio de forma assíncrona.
-   `models/model.py`: Orquestra a lógica de resposta. Carrega o modelo treinado e implementa a cascata de fallbacks (busca direta -> modelo ML -> RAG).
-   `training/train.py`: Script offline para treinar o modelo de classificação Keras e salvar os artefatos.
-   `data/qa_data.py`: Gerencia a base de conhecimento. Carrega pares de pergunta/resposta de um dicionário local, de um cache (`qa_cache.json`) ou de uma API externa (configurada via `QA_API_URL`).
-   `training/rag_index.py`: Ferramenta para ler os arquivos do projeto, dividi-los em pedaços (`chunks`), gerar embeddings vetoriais e construir o índice de busca (`data/index.faiss`).
-   `services/rag_query.py`: Fornece a função `query()` para realizar uma busca de similaridade no índice FAISS e retornar os `chunks` de texto mais relevantes.
-   `core/rag_pipeline.py`: Conecta a busca (RAG) a um Large Language Model (LLM) como o GPT da OpenAI para gerar uma resposta coesa a partir do contexto recuperado.
-   `.gitignore`: Configurado para ignorar ambientes virtuais, caches e artefatos gerados, mantendo o repositório limpo.

---

## Configuração

A configuração do agente é feita principalmente através de variáveis de ambiente, que podem ser definidas em um arquivo `.env` na raiz do projeto.

#### Arquivo `.env`

Crie um arquivo `.env` e preencha com as seguintes variáveis, conforme necessário. O agente lerá este arquivo na inicialização.

```dotenv
# --- Provedor de LLM (Escolha um) ---
# Defina qual LLM usar para a funcionalidade RAG: "openai" ou "gemini".
LLM_PROVIDER="gemini"

# --- OpenAI ---
# Necessário se LLM_PROVIDER for "openai".
OPENAI_API_KEY="sk-SUA_CHAVE_AQUI"
OPENAI_MODEL="gpt-3.5-turbo"

# --- Google Gemini ---
# Necessário se LLM_PROVIDER for "gemini".
GEMINI_API_KEY="sua-chave-gemini-aqui"

# --- API de Dados (Opcional) ---
# URL para carregar perguntas e respostas dinamicamente.
# QA_API_URL="http://exemplo.com/api/qa"
```

#### Detalhes das Variáveis

-   **`LLM_PROVIDER`**: **Obrigatório para RAG.** Define qual provedor de LLM será usado. Valores válidos: `"openai"` ou `"gemini"`. O sistema se adaptará automaticamente à sua escolha.
-   **`OPENAI_API_KEY`**: Chave da API da OpenAI, necessária se `LLM_PROVIDER` for `"openai"`.
-   **`GEMINI_API_KEY`**: Chave da API do Google AI Studio, necessária se `LLM_PROVIDER` for `"gemini"`.
-   **`QA_API_URL`**: (Opcional) URL para especificar uma fonte externa para a base de conhecimento.
-   **`TRANSFORMERS_NO_CUDA=1`**: (Opcional, via terminal) Variável de ambiente útil para forçar o uso de CPU em máquinas sem GPU, evitando erros com `sentence-transformers`.

---

### Estrutura de Diretórios

A estrutura de diretórios foi organizada para separar as responsabilidades e facilitar a manutenção:

-   `core/`: Contém a lógica principal e a orquestração do agente.
    -   `main.py`: Ponto de entrada da aplicação, responsável pelo loop de interação com o usuário e pela interface de linha de comando (CLI).
    -   `rag_pipeline.py`: Orquestra o pipeline de Geração Aumentada por Recuperação (RAG), montando o prompt e consultando o LLM.
    -   `utils.py`: Funções utilitárias compartilhadas.
-   `services/`: Módulos que fornecem serviços específicos.
    -   `get_audio.py`: (Se aplicável) Lógica para captura de áudio.
    -   `rag_query.py`: Serviço para consultar o índice vetorial FAISS e recuperar chunks de texto relevantes.
-   `models/`: Contém os artefatos e a lógica do modelo de Machine Learning.
    -   `model.h5`: O modelo de classificação de intenção treinado.
    -   `tokenizer.json`: O tokenizer para o modelo.
    -   `meta.json`: Metadados associados ao modelo ou ao índice RAG.
    -   `model.py`: Carrega o modelo e o tokenizer, e contém a função `responder()` que encapsula a lógica de decisão.
-   `data/`: Armazena os dados utilizados pelo agente.
    -   `index.faiss`: O índice vetorial para o RAG.
    -   `qa_data.py`: A base de conhecimento principal (perguntas e respostas).
    -   `respostas.json`: Cache das respostas para acesso rápido.
-   `training/`: Scripts para treinamento de modelos e criação de índices.
    -   `train.py`: Script para treinar o modelo de classificação de intenção.
    -   `rag_index.py`: Script para criar o índice FAISS a partir de uma base de código.
-   `tts_cache/`: Diretório de cache para os arquivos de áudio sintetizados.
-   `__pycache__/`: Cache de bytecode do Python.
-   `requirements.txt`: Dependências do projeto.
-   `README.md`: Este guia.