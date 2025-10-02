from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import json, pickle, numpy as np
import nltk, random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Descargar recursos necesarios (solo si no están ya en cache)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = FastAPI()

# Cargar recursos
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

model = load_model('chatbot_model.h5')
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]
    return tokens

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

def predict_class(sentence, threshold=0.35):
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
    for intent in intents['intents']:
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses', ["Lo siento."]))
    return "Lo siento, no encuentro respuesta adecuada."

# ----------- API -----------

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def api_chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        return {"reply": "Escribe algo para comenzar."}
    tag, conf = predict_class(message)
    reply = get_response(tag)
    return {"reply": reply, "tag": tag, "confidence": conf}

# ----------- Ruta raíz opcional (sirve index.html) -----------

@app.get("/")
def index():
    return FileResponse("index.html")
