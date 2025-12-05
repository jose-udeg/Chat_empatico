from datetime import datetime
import streamlit.components.v1 as components
import html as _html
import streamlit as st # Para la interfaz gráfica
import numpy as np # Manejo de arrays
import tensorflow as tf # Para cargar el modelo de red neuronal
import json # Para manejar archivos JSON
from nltk.stem.snowball import SnowballStemmer # Para el stemming en español
import nltk # Para tokenización de las frases
import pickle # Para cargar datos preprocesados
import random # Para respuestas aleatorias
import os

# Configuración de la Página, título y diseño
st.set_page_config(page_title="Avanna - Compañero Emocional", layout="wide")

# INYECCIÓN DE CSS PERSONALIZADO (Diseño y Confianza)
def inject_custom_css():
        st.markdown("""
        <style>
                    
    html, body, .stApp {
        background: #ffffff !important;
    }

    header.stAppHeader, [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }

    [data-testid="stMain"] {
        background-color: #ffffff !important;
    }

    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    section[data-testid="stSidebar"],
    footer {
        background: #ffffff !important;
    }

    :root{
        --bg: #ffffff;
        --card: #ffffff;
        --muted: #6b7280;
        --accent: #4da6ff;
        --accent-2: #ffd1e3;
        --assistant: #fff0f6;
        --user: #ffffff;
        --shadow: 0 6px 18px rgba(32,33,36,0.08);
        --radius: 14px;
    }

    .main > div {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: var(--bg) !important;
        color: #111827 !important;
    }

        .message-row{ display:flex; gap:18px; margin-bottom:12px; align-items:flex-end; }
        .message-row.user { 
            display: flex; 
            
            flex-direction: row-reverse;  
            flex-wrap: wrap; 
            gap: 20px; 
        }

        .message-row.user > div:first-child { 
            display:flex; 
            flex-direction:column; 
            align-items:flex-end; 
            justify-content:center; 
        }
        
        .bubble{ 
            display: flex;             
            align-items: center;       
            max-width:70%; 
            padding: 16px 20px;        
            border-radius:12px; 
            box-shadow: 0 2px 8px rgba(12,12,12,0.04); 
            font-size:15px; 
            line-height:1.4;           
        }

        .bubble.assistant{
            background: linear-gradient(180deg, var(--assistant), #fff);
            border-left:4px solid #ff9ab6; 
            color:#111827;
            border-bottom-left-radius:6px;
        }

        .bubble p {
            margin: 0 !important;
            padding: 0 !important;
        }

        .bubble.force-inline-text {
            white-space: nowrap !important; 
            word-break: normal !important; 
            overflow-wrap: break-word; 
        }

        .bubble.user{
            background: #ffffff; 
            color:#111827; 
            border-bottom-right-radius:6px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            
            padding-left: 20px !important;
            padding-right: 24px !important; 
            box-shadow: 
                4px 0 0 0 var(--accent) inset, 
                0 2px 8px rgba(12,12,12,0.04);            
            display: flex;
            align-items: center;
        }
                    
            .bubble-meta-container {
                display:flex; 
                flex-direction:column; 
                align-items:flex-end;
            }

        .avatar{ width:40px; height:40px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:18px; }
        .avatar.assistant{ background: linear-gradient(45deg, #ff9ab6, #ffd1e3); box-shadow: 0 4px 14px rgba(255,154,182,0.18); }
        .avatar.user{ background: linear-gradient(45deg, #7fd3c7, #4db6ac); color:#fff}
        .meta{ 
            font-size:12px; 
            color:var(--muted); 
            margin-top:6px;
            align-self: flex-end; 
        }

        .input-area{ display:flex; gap:10px; align-items:center; margin-top:12px }
        .input-box{ flex:1; padding:10px 12px; border-radius:10px; border:1px solid rgba(15,23,42,0.06); background: #fff }

        @media (max-width: 768px){ .bubble{ max-width:86% } .chat-window{ height:48vh } }
        </style>
        """, unsafe_allow_html=True)

# Inyectar CSS al inicio de la aplicación
inject_custom_css()

# Inicializar el stemmer (Español)

stemmer = SnowballStemmer('spanish')

# Carga Datos y Modelo
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError: # Manejo de error si no se encuentra el archivo
    st.error("Error: No se encontró 'data.pickle'. Ejecuta train_model.py primero.")
    st.stop()
    
try:
    model = tf.keras.models.load_model('model.h5') # Carga el modelo entrenado
except OSError:
    st.error("Error: No se encontró 'model.h5'. Ejecuta train_model.py primero.")
    st.stop()

with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

# --- Funciones ---

def bag_of_words(s, words): # Crear la bolsa de palabras para la frase de entrada
    bag = [0 for _ in range(len(words))] #
    s_words = nltk.word_tokenize(s, language='spanish') # Tokeniza la frase de entrada en español
    s_words = [stemmer.stem(word.lower()) for word in s_words if word != "?"] # Stemming y limpieza

    for se in s_words:
        for i, w in enumerate(words): # Recorre las palabras conocidas, por medio de comparación
            if w == se:
                bag[i] = 1 # Si la palabra está presente, marca con 1
    return np.array(bag)

def chat(inp):
# Predecir la respuesta basada en la entrada del usuario
    bow = bag_of_words(inp, words)
    results = model.predict(np.array([bow]), verbose=0)[0] # Obtiene las probabilidades de cada etiqueta
    results_index = np.argmax(results) # Índice de la etiqueta con mayor probabilidad
    tag = labels[results_index] # Etiqueta correspondiente, basada en el índice
    probabilidad = results[results_index] # Probabilidad asociada a la etiqueta
    
    # Debug para mostrar la confianza de la predicción
    # ------------------------------------------------
    # En letra chica, debajo del mensaje
    if probabilidad > 0.7:
        st.caption(f"✅ Entendido como: '{tag}' (Confianza: {probabilidad:.2%})")
    else:
        st.error(f"⚠️ Confianza baja: {probabilidad:.2%} (Intentó clasificar como: {tag})")
    # ------------------------------------------------

    # Umbral de Confianza
    #--------------------------------
    if probabilidad > 0.7: # Si la confianza es alta suficiente
        for tg in data['intents']:
            if tg['tag'] == tag:
                return random.choice(tg['responses']) # Respuesta aleatoria de la etiqueta correspondiente
    
    # Frases desconocidas
    else:
        # Guardar la frase desconocida
        with open("missed_queries.txt", "a", encoding='utf-8') as f:
            f.write(inp + "\n")
            
        return "Siento que no estoy entendiendo del todo. ¿Podrías intentar explicármelo con otras palabras?"

# --- Interfaz Gráfica (Amigable) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=60) # Ícono visual de apoyo
    st.title("💖 Avanna: Tu Compañero de Bienestar")
    st.markdown("---")
    
    st.header("🔒 Un Espacio Seguro")
    st.info("Todo lo que compartas aquí es **confidencial**. Avanna no es una terapeuta, sino una **amiga virtual** que te acompaña.")
    
    st.header("✨ ¿Cómo Funciona?")
    st.caption("Escribes cómo te sientes, y Avanna intentará entender tu emoción y responder con comprensión y sugerencias de apoyo.")
    
    st.markdown("---")
    st.caption("*Avanna v2.1. Diseñado para generar confianza y un ambiente de calma.*")

# --- Interfaz Principal (Diseño Visual) ---

# Título enmarcado con degradado (MODIFICACIÓN DE DISEÑO)
with st.container(border=True):
    st.header("💬 Hablemos de lo que sientes")
    st.markdown("Hola, soy Avanna. Estoy aquí para escucharte **sin juicio** y con respeto. Tómate tu tiempo y dime: **¿Qué hay en tu mente?**")
    
st.divider()

def _build_bubble_html(role, content, ts, allow_html=False):
    # 1. Normalizar y escapar el contenido para seguridad (XSS)
    content_safe = ' '.join(content.replace('\r', ' ').replace('\n', ' ').split())
    if not allow_html:
        # Escapamos todo el contenido
        content_final = _html.escape(content_safe)
    else:
        content_final = content_safe 
    
    content = content_final

    avatar = '💖' if role == 'assistant' else '👤'
    row_class = 'message-row assistant' if role == 'assistant' else 'message-row user'
    bubble_class = 'bubble assistant' if role == 'assistant' else 'bubble user'
    
    # Inyectamos estilos en línea en la burbuja del USUARIO
    if role == 'user':
        # Añadimos un estilo inline que fuerza el texto a no romperse
        bubble_class += " force-inline-text"

    if role == 'assistant':
        return f"<div class='{row_class}'><div class='avatar assistant'>{avatar}</div><div><div class='{bubble_class}'>{content}</div><div class='meta'>{ts}</div></div></div>"
    else:
        return f"<div class='{row_class}'><div class='{bubble_class}'>{content}</div><div class='meta'>{ts}</div></div><div class='avatar user'>{avatar}</div></div>"

if 'chat_body' not in st.session_state:
    st.session_state.chat_body = []

if 'chat_placeholder' not in st.session_state:
    st.session_state.chat_placeholder = st.empty()

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenido/a. ¿Cómo te sientes en este momento? Estoy aquí para ti.", "time": datetime.now().strftime('%H:%M')}]

if st.session_state.messages and not st.session_state.chat_body:
    for msg in st.session_state.messages:
        st.session_state.chat_body.append(_build_bubble_html(msg['role'], msg['content'], msg.get('time', '')))

def _render_chat_placeholder():
    prefix = '<div class="chat-card"><div class="chat-window">'
    suffix = '</div></div>'
    html = prefix + ''.join(st.session_state.chat_body) + '<div id="end"></div>' + suffix
    st.session_state.chat_placeholder.markdown(html, unsafe_allow_html=True)

# Initial render
_render_chat_placeholder()

# Entrada de chat 
if prompt := st.chat_input("¿Cómo te sientes hoy? Estoy lista para escucharte..."):
    now = datetime.now().strftime('%H:%M')
    user_html = _build_bubble_html('user', prompt, now)
    st.session_state.chat_body.append(user_html)
    _render_chat_placeholder()

    typing_id = len(st.session_state.chat_body)
    typing_html = _build_bubble_html('assistant', '<i>Escribiendo...</i>', '')
    st.session_state.chat_body.append(typing_html)
    _render_chat_placeholder()

    response = chat(prompt)
    assistant_html = _build_bubble_html('assistant', response, datetime.now().strftime('%H:%M'))
    st.session_state.chat_body[typing_id] = assistant_html
    st.session_state.messages.append({"role": "user", "content": prompt, "time": now})
    st.session_state.messages.append({"role": "assistant", "content": response, "time": datetime.now().strftime('%H:%M')})
    _render_chat_placeholder()