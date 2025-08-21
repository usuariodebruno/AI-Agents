"""
Funções utilitárias para o modelo e treinamento.
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def criar_tokenizer(textos: dict) -> Tokenizer:
    """Cria e treina um tokenizer a partir de um dicionário de textos."""
    t = Tokenizer(num_words=None, oov_token="<unk>")
    t.fit_on_texts(textos.keys())
    return t

def texto_para_sequencia(tokenizer: Tokenizer, texto: str, max_len: int = 10) -> np.ndarray:
    """Converte um texto em uma sequência de tokens e aplica padding."""
    seq = tokenizer.texts_to_sequences([texto])
    return pad_sequences(seq, maxlen=max_len)
