import streamlit as st
import numpy as np
import tensorflow as tf
import json
from nltk.stem.snowball import SnowballStemmer
import nltk
import pickle
import random
import os

# --- Configuración de la Página ---
st.set_page_config(page_title="Avanna - Compañero Emocional", layout="wide")

# Inicializar el stemmer (Español)
stemmer = SnowballStemmer('spanish')

# --- Cargar Datos y Modelo ---
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    st.error("Error: No se encontró 'data.pickle'. Ejecuta train_model.py primero.")
    st.stop()
    
try:
    model = tf.keras.models.load_model('model.h5')
except OSError:
    st.error("Error: No se encontró 'model.h5'. Ejecuta train_model.py primero.")
    st.stop()

with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

# --- Funciones ---

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s, language='spanish')
    s_words = [stemmer.stem(word.lower()) for word in s_words if word != "?"]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat(inp):
    # 1. Predecir
    bow = bag_of_words(inp, words)
    results = model.predict(np.array([bow]), verbose=0)[0] 
    results_index = np.argmax(results)
    tag = labels[results_index]
    probabilidad = results[results_index]
    
    # --- DEBUG: Mostrar qué está pensando el robot ---
    # Esto aparecerá en letra pequeña debajo de tu mensaje
    if probabilidad > 0.7:
        st.caption(f"✅ Entendido como: '{tag}' (Confianza: {probabilidad:.2%})")
    else:
        st.error(f"⚠️ Confianza baja: {probabilidad:.2%} (Intentó clasificar como: {tag})")
    # ------------------------------------------------

    # 2. Umbral de Confianza
    if probabilidad > 0.7:
        for tg in data['intents']:
            if tg['tag'] == tag:
                return random.choice(tg['responses'])
    
    # 3. Manejo de lo desconocido
    else:
        # Guardar la frase desconocida
        with open("missed_queries.txt", "a", encoding='utf-8') as f:
            f.write(inp + "\n")
            
        return "Siento que no estoy entendiendo del todo. ¿Podrías intentar explicármelo con otras palabras?"

# --- Interfaz Gráfica (Amigable) ---
with st.sidebar:
    st.title("Sobre Avanna")
    st.write("Soy una inteligencia artificial en desarrollo, aprendiendo a ser más empática cada día.")
    st.info("Este es un espacio seguro para desahogarte y hablar libremente de lo que sientas.")
    st.markdown("---")
    st.caption("Avanna v2.1 - Tu amiga virtual")

# --- Interfaz Principal ---
st.title("Habla con Avanna 🤖")
st.markdown("Hola, soy Avanna. Estoy aquí para escucharte como una amiga.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de chat
if prompt := st.chat_input("¿Cómo te sientes hoy?"):
    # Usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # IA
    response = chat(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)