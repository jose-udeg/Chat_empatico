import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from nltk.stem.snowball import SnowballStemmer
import nltk
import pickle
import random

# Inicializar stemmer
stemmer = SnowballStemmer('spanish')

# --- Cargar y Preparar Datos ---
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern, language='spanish')
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []
out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1) if w in wrds else bag.append(0)

    output_row = list(out_empty)
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# --- Construcción del Modelo (MEJORADA) ---
model = Sequential()

# Capa de entrada más grande (128 neuronas) para captar mejor los patrones
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))

# Capa intermedia de refuerzo (64 neuronas)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(len(output[0]), activation='softmax'))

# Compilación
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping con un poco más de paciencia
callback = EarlyStopping(monitor='accuracy', patience=50, restore_best_weights=True)

print("--- Entrenando Modelo Potenciado ---")
history = model.fit(training, output, epochs=500, batch_size=5, verbose=1, callbacks=[callback])

model.save("model.h5", history)
print("--- Modelo guardado correctamente ---")