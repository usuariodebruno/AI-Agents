"""
Arquivo para carregar perguntas e respostas (QA) de uma API ou cache local.
Se a API não estiver disponível, usa dados embutidos como fallback.
"""

import os
import json
import urllib.request
import urllib.error

# Arquivo de cache local
_CACHE_FILE = "qa_cache.json"
# Pode configurar a URL da API via variável de ambiente QA_API_URL
_API_URL = os.environ.get("QA_API_URL") 

# Dados padrão embutidos (fallback)
_default_qa = {
    "qual é o seu nome": "Eu sou seu assistente de IA.",
    "qual seu nome": "Eu sou seu assistente de IA.",
    "como você está": "Estou bem, obrigado por perguntar!",
    "como vc está": "Estou bem, obrigado por perguntar!",
    "o que você faz": "Eu respondo perguntas básicas usando IA.",
    "como você pode me ajudar": "Posso responder perguntas simples e conversar sobre tópicos básicos.",
    "o que você sabe fazer": "Posso responder perguntas pré-definidas, falar e ouvir (se integrado a áudio).",
    "quem te criou": "Fui criado por você (ou pelo desenvolvedor do projeto).",
    "onde você mora": "Eu existo como um programa, então não tenho um lugar físico.",
    "quantos anos você tem": "Eu não tenho idade como humanos; sou um programa de computador.",
    "me diga uma piada": "Por que o programador foi ao médico? Porque tinha um bug!",
    "bom dia": "Bom dia! Como posso ajudar?",
    "boa tarde": "Boa tarde! Em que posso ajudar?",
    "boa noite": "Boa noite! Precisa de algo antes de descansar?",
    "obrigado": "De nada — estou aqui para ajudar!",
    "valeu": "Por nada!",
    "tchau": "Até mais!",
    "adeus": "Até logo — volte sempre!",
    "o que você pode fazer por mim": "Posso responder perguntas básicas do banco de dados e gerar respostas de voz se estiver configurado.",
    "como usar": "Digite uma pergunta e eu tentarei responder com base no meu conjunto de dados.",
    "qual é a data de hoje": "Desculpe, não tenho acesso ao relógio neste modo, mas você pode verificar a data no seu sistema.",
    "que horas são": "Não consigo acessar o relógio agora, verifique o relógio do seu sistema.",
    "me ajude": "Diga qual é a sua dúvida e tentarei responder com os recursos que tenho.",
    "repita": "Claro — o que você quer que eu repita?"
}


def _load_cache():
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def _save_cache(data: dict):
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _fetch_from_api(url: str, timeout: int = 5) -> dict:
    """Tenta buscar JSON da API. Formato aceito: dicionário simples {pergunta: resposta, ...}
    ou objeto com chave 'qa_pairs'."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IA-BASE/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                # se o JSON tem chave 'qa_pairs', usar esse valor
                if "qa_pairs" in data and isinstance(data["qa_pairs"], dict):
                    return data["qa_pairs"]
                # caso contrário assumir que é o próprio dict de pares
                return data
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        pass
    return None


def refresh_qa(url: str = None) -> dict:
    """Atualiza o dicionário de perguntas/respostas a partir da API (se disponível).
    Retorna o dicionário usado (API, cache ou fallback)."""
    u = url or _API_URL
    if u:
        fetched = _fetch_from_api(u)
        if fetched and isinstance(fetched, dict) and len(fetched) > 0:
            _save_cache(fetched)
            return fetched
    # tentar cache local
    cached = _load_cache()
    if cached:
        return cached
    # fallback para dados embutidos
    return _default_qa


# Carregar inicialmente: tentar API -> cache -> default
qa_pairs = refresh_qa()

# Expor função para o restante do código reconsultar
__all__ = ["qa_pairs", "refresh_qa"]
