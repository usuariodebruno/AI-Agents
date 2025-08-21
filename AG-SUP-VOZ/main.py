"""
# Agente IA com síntese de voz
# Este script combina um modelo de IA para responder perguntas
# com síntese de voz usando gTTS ou edge-tts.
"""
from dotenv import load_dotenv
load_dotenv()

from gtts import gTTS
from model import responder, respostas
import threading
import queue
import os
import shutil
import tempfile
import subprocess
import hashlib

# tentar edge-tts para voz neural mais natural
try:
    import asyncio
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    print("INFO: Usando edge-tts para síntese de voz.")
except Exception:
    EDGE_TTS_AVAILABLE = False
    print("AVISO: edge-tts não encontrado, tentando fallback.")

# fallback para pyttsx3 se edge-tts não estiver disponível
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    print("INFO: pyttsx3 encontrado como fallback.")
except Exception:
    PYTTSX3_AVAILABLE = False
    print("ERRO: Nenhum motor de TTS (edge-tts, pyttsx3) disponível.")

# escolher player disponível para reproduzir mp3 gerado (se necessário)
_player = None
for cmd in ("mpg123", "ffplay", "afplay", "mpv", "vlc"):
    if shutil.which(cmd):
        _player = cmd
        print(f"INFO: Reprodutor de áudio encontrado: {_player}")
        break

if not _player:
    print("AVISO: Nenhum reprodutor de áudio (mpg123, ffplay, etc.) encontrado. A reprodução pode falhar.")

# inicializar pyttsx3 uma vez (fallback)
_engine = None
if PYTTSX3_AVAILABLE and not EDGE_TTS_AVAILABLE:
    print("INFO: Inicializando motor pyttsx3...")
    _engine = pyttsx3.init()
    for voice in _engine.getProperty('voices'):
        if 'pt' in voice.id.lower() or 'portuguese' in voice.name.lower():
            _engine.setProperty('voice', voice.id)
            break
    _engine.setProperty('rate', 160)
    print("INFO: Motor pyttsx3 inicializado.")

_speech_queue = queue.Queue()

CACHE_DIR = "tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path_for_text(text: str) -> str:
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.mp3")

def _play_file_with_player(path: str):
    if _player is None:
        # no player found, try default OS open
        try:
            if os.name == 'posix':
                subprocess.run(["xdg-open", path], check=False)
            elif os.name == 'nt':
                os.startfile(path)
            else:
                subprocess.run(["open", path], check=False)
        except Exception:
            pass
        return

    if _player == 'ffplay':
        # ffplay prints a lot; use -autoexit -nodisp
        cmd = [_player, '-nodisp', '-autoexit', '-loglevel', 'quiet', path]
    elif _player == 'vlc':
        cmd = [_player, '--intf', 'dummy', path]
    else:
        cmd = [_player, path]

    try:
        print(f"INFO: Executando comando de áudio: {' '.join(cmd)}")
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        print(f"ERRO: Falha ao executar o reprodutor de áudio: {e}")

async def _edge_tts_save(text: str, path: str, voice: str = "pt-BR-FranciscaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

def _synthesize_with_edge(text: str) -> str:
    # retorna caminho do arquivo mp3 gerado
    fd, path = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_edge_tts_save(text, path))
        loop.close()
        return path
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        raise

# pré-sintetizar respostas conhecidas para reduzir latência
if EDGE_TTS_AVAILABLE:
    for r in set(respostas):
        path = _cache_path_for_text(r)
        if not os.path.exists(path):
            try:
                mp3 = _synthesize_with_edge(r)
                # mover arquivo gerado para cache
                try:
                    os.replace(mp3, path)
                except Exception:
                    try:
                        os.remove(path)
                        os.replace(mp3, path)
                    except Exception:
                        pass
            except Exception as e:
                print("Falha ao pré-sintetizar resposta:", e)

def _speech_worker():
    while True:
        texto = _speech_queue.get()
        try:
            cached = _cache_path_for_text(texto)
            if os.path.exists(cached):
                _play_file_with_player(cached)
            else:
                if EDGE_TTS_AVAILABLE:
                    try:
                        mp3_path = _synthesize_with_edge(texto)
                        # mover para cache
                        try:
                            os.replace(mp3_path, cached)
                            _play_file_with_player(cached)
                        except Exception as e_replace:
                            print(f"AVISO: Falha ao mover áudio para o cache: {e_replace}")
                            _play_file_with_player(mp3_path)
                            try:
                                os.remove(mp3_path)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"ERRO: Falha na síntese com edge-tts: {e}")
                        if _engine:
                            print("INFO: Tentando fallback para pyttsx3...")
                            _engine.say(texto)
                            _engine.runAndWait()
                        else:
                            print("ERRO TTS: Nenhum motor de fallback disponível.", e)
                else:
                    if _engine:
                        print("INFO: Usando pyttsx3 para falar.")
                        _engine.say(texto)
                        _engine.runAndWait()
                    else:
                        print("Resposta (sem áudio):", texto)
        except Exception as e:
            print("Erro ao reproduzir áudio:", e)
        _speech_queue.task_done()

_thread = threading.Thread(target=_speech_worker, daemon=True)
_thread.start()

def falar(texto: str):
    """Enfileira texto para reprodução assíncrona (retorna imediatamente)."""
    _speech_queue.put(texto)

def main():
    service_name = os.environ.get("SERVICE_NAME", "este projeto")
    mensagem_boas_vindas = f"Olá, no que posso ajudar sobre o {service_name}?"
    
    print(mensagem_boas_vindas)
    falar(mensagem_boas_vindas)
    
    while True:
        texto_usuario = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        
        if texto_usuario.lower() in ["sair", "exit", "quit"]:
            print("Encerrando...")
            break
        
        resposta, sugestoes = responder(texto_usuario)
        print("Resposta:", resposta)
        falar(resposta)  # Fala apenas a resposta principal

        if sugestoes:
            print("\nSugestões para aprofundar:")
            sugestoes_texto = "Aqui estão algumas sugestões para aprofundar: "
            for i, sug in enumerate(sugestoes):
                print(f"- {sug}")
                sugestoes_texto += f" {i+1}: {sug}." # Constrói a frase para ser falada
            
            falar(sugestoes_texto) # Enfileira as sugestões para serem faladas
            print() # Adiciona uma linha em branco para espaçamento

if __name__ == "__main__":
    main()
