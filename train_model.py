import json # Para manipular archivos JSON.
import numpy as np # Para manejar arreglos numéricos.
import tensorflow as tf # Para construir y entrenar el modelo de red neuronal.
from tensorflow.keras.models import Sequential # Para construir el modelo secuencial.
from tensorflow.keras.layers import Dense, Dropout # Para permitir capas densas y para evitar el sobreajuste.
from tensorflow.keras.callbacks import EarlyStopping # Para detener el entrenamiento si no hay mejora.
from nltk.stem.snowball import SnowballStemmer # Reduce la variabilidad de las palabras. Apoya en el uso del español.
import nltk # Para el procesamiento de lenguaje natural, realizar tokenización.
import pickle # Para guardar datos preprocesados.

# Inicializar stemmer
stemmer = SnowballStemmer('spanish')

# Se abre el archivo JSON que contiene los datos de entrenamiento
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

words = [] # Lista de todas las palabras unicas
labels = [] # Lista de etiquetas (intenciones)
docs_x = [] # Frases tokenizadas
docs_y = [] # Etiquetas correspondientes

for intent in data['intents']: #itera a través de cada intención
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern, language='spanish') # Tokeniza la frase
        words.extend(wrds) #adjunta  las palabras a la lista de palabras general
        docs_x.append(wrds) #Agrega la frase tokenizada a docs_x
        docs_y.append(intent['tag']) #relaciona la frase con su etiqueta

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] # Stemming y limpieza, convirtiendo a minúsculas y eliminando signos de interrogación
words = sorted(list(set(words))) # Elimina duplicados y ordena
labels = sorted(labels) # Ordena las etiquetas

# Creacion de la bag of words

# Inicializamos las listas que contendrán los datos de entrenamiento
training = []
output = []
out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x): # Recorremos cada frase tokenizada
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc] 

    for w in words: #identificamos si la palabra está en la frase
        bag.append(1) if w in wrds else bag.append(0)

    output_row = list(out_empty) # Creamos una fila de salida con ceros
    output_row[labels.index(docs_y[x])] = 1 # Marcamos la etiqueta correspondiente con un 1

    training.append(bag) 
    output.append(output_row)

# Convertimos a array de numpy para facilitar el manejo con TensorFlow
training = np.array(training) 
output = np.array(output)

# Guardamos los datos preprocesados en un archivo pickle (binario)
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Construcción del modelo de red neuronal

model = Sequential() #Inicializamos el modelo secuencial

# Capa de entrada, con 128 neuronas. Define el tamaño de la entrada y usa ReLU como función de activación
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5)) # Dropout para evitar sobreajuste

# Capa media, con 64 neuronas y ReLU. Mismamente Dropout para evitar sobreajuste
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida, con tantas neuronas como etiquetas y softmax para obtener probabilidades
model.add(Dense(len(output[0]), activation='softmax'))

# Compilación del modelo usando Adam y categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configuración de Early Stopping para evitar sobreentrenamiento
callback = EarlyStopping(monitor='accuracy', patience=50, restore_best_weights=True)

# Entrenamiento del modelo, con 500 épocas y batch size de 5
print("--- Entrenando Modelo Potenciado ---")
history = model.fit(training, output, epochs=500, batch_size=5, verbose=1, callbacks=[callback])

# Guardado del modelo entrenado
model.save("model.h5", history)
print("--- Modelo guardado correctamente ---")