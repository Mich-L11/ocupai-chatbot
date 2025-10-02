# train_chatbot.py
import json
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# Usamos TensorFlow.keras (más portable)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Descargar recursos de NLTK (solo la primera vez; si ya los descargaste, no dañará)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Cargar intents.json
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',', ';', ':']

# Procesamiento de patrones
for intent in intents['intents']:
    tag = intent.get('tag')
    patterns = intent.get('patterns', [])
    for pattern in patterns:
        # tokenizar
        tokens = nltk.word_tokenize(pattern)
        # lematizar y pasar a minúsculas
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens if t not in ignore_letters]
        words.extend(tokens)
        documents.append((tokens, tag))
    if tag not in classes:
        classes.append(tag)

# Ordenar y eliminar duplicados
words = sorted(set(words))
classes = sorted(set(classes))

# Guardar vocabulario y clases
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparar datos de entrenamiento (bag-of-words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    token_list, tag = doc
    bag = [1 if w in token_list else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    training.append((bag, output_row))

random.shuffle(training)

train_x = np.array([t[0] for t in training], dtype=np.float32)
train_y = np.array([t[1] for t in training], dtype=np.float32)

# Construcción del modelo
model = Sequential()
model.add(Dense(250, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar
history = model.fit(train_x, train_y, epochs=2000, batch_size=5, verbose=1)

# Guardar modelo y historial
model.save('chatbot_model.h5')
with open('train_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Entrenamiento completado. Archivos guardados: chatbot_model.h5, words.pkl, classes.pkl, train_history.pkl")
