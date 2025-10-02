# chatbot.py
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Si no has descargado los recursos con anterioridad, el siguiente bloque los descarga.
# En la mayoría de casos solo es necesario la primera vez.
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Cargar archivos generados en el entrenamiento
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

model = load_model('chatbot_model.h5')

# Normaliza y lematiza la oración ingresada
def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]
    return tokens

# Convierte a vector bag-of-words
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

# Predice la etiqueta (si la probabilidad máxima es baja, devuelve None)
def predict_class(sentence, model, words, classes, thresh=0.40):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]  # array de probabilidades
    max_prob = float(np.max(res))
    if max_prob < thresh:
        return None, max_prob
    max_index = int(np.argmax(res))
    return classes[max_index], max_prob

# Devuelve una respuesta aleatoria del intents.json
def get_response(tag, intents_json):
    if tag is None:
        return "Lo siento, no estoy seguro de haber entendido. ¿Podrías reformular o escoger una opción del menú?"
    for intent in intents_json['intents']:
        if intent.get('tag') == tag:
            return np.random.choice(intent.get('responses', ["Lo siento, no tengo una respuesta para eso."]))
    return "Lo siento, no encuentro una respuesta adecuada."

# Bucle interactivo
def main():
    print("OcupAI BOT listo. Escribe 'salir' para terminar.\n")
    while True:
        message = input("Tú: ").strip()
        if not message:
            continue
        if message.lower() in ['salir', 'exit', 'chao', 'adiós', 'adios']:
            print("Chatbot: ¡Hasta luego!")
            break
        tag, confidence = predict_class(message, model, words, classes, thresh=0.40)
        # debug: print("DEBUG:", tag, confidence)
        response = get_response(tag, intents)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
