import json
import numpy as np
import tensorflow as tf
import nltk
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("punkt")
nltk.download("wordnet")

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []

# Tokenize words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent["tag"]))
    classes.append(intent["tag"])

words = sorted(set(lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()))
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
X_train, y_train = np.array([i[0] for i in training]), np.array([i[1] for i in training])

# Build model
model = Sequential([
    Dense(128, activation="relu", input_shape=(len(X_train[0]),)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(y_train[0]), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)

# Save model
model.save("chatbot_model.h5")
np.save("words.npy", words)
np.save("classes.npy", classes)

print("Model trained and saved successfully!")