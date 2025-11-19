import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.stem.lancaster import LancasterStemmer
import nltk
import pickle
import random

# Inicializar el stemmer (para reducir palabras a su raíz)
# nltk.download('punkt') # Descomenta y ejecuta si es la primera vez
stemmer = LancasterStemmer()

# --- Cargar y Preparar Datos ---
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

words = []
labels = []
docs_x = [] # Patrones
docs_y = [] # Tags (Intenciones)

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenización: Divide la frase en palabras
        wrds = nltk.word_tokenize(pattern, language='spanish')
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stemming y limpieza (pasar a minúsculas, quitar duplicados)
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# --- Crear Datos de Entrenamiento (Bag of Words) ---
training = []
output = []
out_empty = [0] * len(labels) # Vector de salida vacío (one-hot encoding)

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    # Crear la "Bolsa de Palabras" (Bag of Words)
    for w in words:
        bag.append(1) if w in wrds else bag.append(0)

    # Crear la etiqueta (One-Hot Encoding)
    output_row = list(out_empty)
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# --- Guardar datos pre-procesados y listas clave ---
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# --- Construcción del Modelo con Funciones de Activación ---

# Capa de entrada: # de palabras únicas
# Capas ocultas: 8 neuronas
# Capa de salida: # de etiquetas únicas

model = Sequential()
# Capa Oculta: Usando la función de activación ReLU (REQUISITO CUMPLIDO)
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
# Capa de Salida: Usando la función de activación Softmax (REQUISITO CUMPLIDO)
model.add(Dense(len(output[0]), activation='softmax')) 

# Compilación y Entrenamiento
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("--- Entrenando Modelo ---")
history = model.fit(training, output, epochs=200, batch_size=8, verbose=1)

# Guardar el modelo entrenado
model.save("model.h5", history)
print("--- Modelo entrenado y guardado en model.h5 ---")