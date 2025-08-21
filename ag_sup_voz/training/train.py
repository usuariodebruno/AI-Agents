"""
Arquivo para treinar um modelo de linguagem simples
usando perguntas e respostas (QA) pré-definidas e salvar os artefatos necessários
"""

import json
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_data import qa_pairs
from core.utils import criar_tokenizer, texto_para_sequencia

# Preparar tokenizer e dados
tokenizer = criar_tokenizer(qa_pairs)
perguntas = list(qa_pairs.keys())
respostas = list(qa_pairs.values())
X = np.array([texto_para_sequencia(tokenizer, p)[0] for p in perguntas])
y = np.arange(len(respostas))

# Modelo simples (mesma arquitetura usada em model.py)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(respostas), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Iniciando treino do modelo...")
model.fit(X, y, epochs=200, verbose=1)

# Salvar artefatos
print("Salvando modelo e tokenizer...")
model.save("models/model.h5")
with open("models/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())
with open("data/respostas.json", "w", encoding="utf-8") as f:
    json.dump(respostas, f, ensure_ascii=False)

print("Treino concluído. Arquivos gerados: models/model.h5, models/tokenizer.json, data/respostas.json")
