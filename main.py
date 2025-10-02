import os
import json
import pickle
import random
import numpy as np
import nltk

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Descargar recursos NLTK si no están
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Inicializar app ---
app = FastAPI()

# Servir archivos estáticos (CSS, JS, imágenes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Variables globales ---
model = None
words = []
classes = []
intents = {}

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# --- Cargar archivos de recursos ---
def load_resources():
    global intents, words, classes, model
    try:
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        print("✅ intents.json cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando intents.json: {e}")

    try:
        with open('words.pkl', 'rb') as f:
            words = pickle.load(f)
        print("✅ words.pkl cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando words.pkl: {e}")

    try:
        with open('classes.pkl', 'rb') as f:
            classes = pickle.load(f)
        print("✅ classes.pkl cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando classes.pkl: {e}")

    try:
        from tensorflow.keras.models import load_model
        model = load_model('chatbot_model.h5')
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando chatbot_model.h5: {e}")
        model = None

load_resources()

# --- Funciones del chatbot ---
def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]
    return tokens

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

def predict_class(sentence, threshold=0.35):
    if model is None:
        return None, 0.0
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    max_prob = float(np.max(res))
    if max_prob < threshold:
        return None, max_prob
    idx = int(np.argmax(res))
    return classes[idx], max_prob

def get_response(tag):
    if tag is None:
        return "Lo siento, no entiendo. ¿Podrías reformular o escoger una opción del menú?"
    for intent in intents.get('intents', []):
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses', ["Lo siento."]))
    return "Lo siento, no encuentro respuesta adecuada."

# --- API ---
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def api_chat(request: ChatRequest):
    if model is None:
        return {"reply": "⚠️ Error: el modelo no está cargado en el servidor."}

    message = request.message.strip()
    if not message:
        return {"reply": "Escribe algo para comenzar."}

    try:
        tag, conf = predict_class(message)
        reply = get_response(tag)
        return {"reply": reply, "tag": tag, "confidence": conf}
    except Exception as e:
        print(f"❌ Error procesando mensaje: {e}")
        return {"reply": "⚠️ Ocurrió un error en el servidor."}

# --- Ruta raíz ---
@app.get("/", response_class=HTMLResponse)
def index():
    try:
        return FileResponse("index.html")
    except:
        return HTMLResponse("<h1>⚠️ No se encontró index.html</h1>")

# --- Habilitar CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
