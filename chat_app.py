import streamlit as st
import numpy as np
import tensorflow as tf
import json
from nltk.stem.lancaster import LancasterStemmer
import nltk
import pickle
import random

# Inicializar el stemmer y cargar los datos
stemmer = LancasterStemmer()

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    st.error("Error: No se encontró el archivo 'data.pickle'. Asegúrate de haber ejecutado 'train_model.py'.")
    st.stop()
    
# Cargar el modelo entrenado
try:
    model = tf.keras.models.load_model('model.h5')
except:
    st.error("Error: No se encontró el archivo 'model.h5'. Asegúrate de que el entrenamiento haya finalizado con éxito.")
    st.stop()


# Cargar los Intents (respuestas)
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

# --- Funciones de Pre-procesamiento y Predicción ---

# Función para crear la Bolsa de Palabras (igual que en train_model.py)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s, language='spanish')
    s_words = [stemmer.stem(word.lower()) for word in s_words if word != "?"]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

# Función principal para la conversación
def chat(inp):
    # Convertir la entrada a Bag of Words
    bow = bag_of_words(inp, words)
    
    # Predecir la Intención
    # La predicción es un array de probabilidades, una por cada etiqueta (Intención)
    results = model.predict(np.array([bow]))[0] 
    
    # Obtener el índice con la probabilidad más alta
    results_index = np.argmax(results)
    
    # Obtener la etiqueta (tag) predicha
    tag = labels[results_index]
    
    # Verificar la confianza (si la probabilidad es muy baja, es una respuesta desconocida)
    # 0.70 (70%) es un buen umbral para empezar.
    if results[results_index] > 0.30:
        for tg in data['intents']:
            if tg['tag'] == tag:
                # Seleccionar una respuesta aleatoria de las respuestas definidas para esa Intención
                response = random.choice(tg['responses'])
                return response, tag # Devolvemos la respuesta y la intención predicha
    else:
        # Respuesta de "No sé" o "Desconocido"
        return "Disculpa, no estoy seguro de cómo responder a eso. ¿Podrías reformularlo?", "desconocido"
    


    # --- Configuración de Streamlit ---
st.set_page_config(page_title="Avanna - Compañero Emocional", layout="wide")

st.title("Avanna 🧘 Compañero Emocional IA")
st.markdown("Soy Avanna, estoy aquí para escucharte y ayudarte a validar tus sentimientos.")

# Inicializar el historial de chat en el estado de sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Lógica de la Entrada del Usuario ---
if prompt := st.chat_input("Dime, ¿cómo te sientes hoy?"):
    
    # 1. Añadir el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar la respuesta de la IA
    response, tag_predicho = chat(prompt)
    
    # 3. Mostrar la respuesta de la IA
    with st.chat_message("assistant"):
        st.markdown(response)
        
        # Opcional: Mostrar la intención predicha para propósitos de prueba
        # st.caption(f"Intención predicha: {tag_predicho}") 

    # 4. Añadir el mensaje de la IA al historial
    st.session_state.messages.append({"role": "assistant", "content": response})