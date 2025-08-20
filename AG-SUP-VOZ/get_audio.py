""" Arquivo para reconhecimento de voz e síntese de fala
    usando SpeechRecognition e pyttsx3.
"""

import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
engine = pyttsx3.init()

while True:
    with sr.Microphone() as source:
        print("Diga algo (ou 'sair' para encerrar):")
        audio = r.listen(source)

        try:
            texto = r.recognize_google(audio, language="pt-BR")
            print("Você disse:", texto)

            if "sair" in texto.lower():
                print("Encerrando...")
                break

            # Resposta simples
            resposta = f"Você disse: {texto}"
            engine.say(resposta)
            engine.runAndWait()

        except sr.UnknownValueError:
            print("Não entendi o que você disse.")
        except sr.RequestError as e:
            print("Erro ao acessar o serviço; {0}".format(e))
