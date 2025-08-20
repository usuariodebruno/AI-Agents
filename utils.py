"""
Contém funções auxiliares para manipulação de texto e criação de tokenizers.
"""
import tensorflow as tf

def criar_tokenizer(qa_pairs):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(list(qa_pairs.keys()))
    return tokenizer

def texto_para_sequencia(tokenizer, texto, max_len=10):
    seq = tokenizer.texts_to_sequences([texto])
    return tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
