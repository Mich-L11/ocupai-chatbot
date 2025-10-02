from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json, pickle, numpy as np
import nltk, random, traceback
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Cargar recursos ---
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)

    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    model = load_model('chatbot_model.h5')
    lemmatizer = WordNetLemmatizer()
    print("✅ Modelo y recursos cargados correctamente")

except Exception as e:
    print("Error cargando modelo o recursos:", e)
    traceback.print_exc()

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
        return "Lo siento, no entiendo. ¿Podrías reformular?"
    for intent in intents['intents']:
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses', ["Lo siento."]))
    return "Lo siento, no encuentro respuesta adecuada."

# ----------- API -----------

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    try:
        message = request.message.strip()
        if not message:
            return {"reply": "Escribe algo para comenzar."}
        tag, conf = predict_class(message)
        reply = get_response(tag)
        return {"reply": reply, "tag": tag, "confidence": conf}
    except Exception as e:
        print("Error en /api/chat:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )

# ----------- Ruta raíz (sirve index.html) -----------

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("index.html")

# === Habilitar CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
