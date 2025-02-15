from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

# Load intents JSON
with open("intents.json", "r") as file:
    intents = json.load(file)

# Load trained chatbot model
model = tf.keras.models.load_model("chatbot_model.h5")

# Preprocess input
lemmatizer = WordNetLemmatizer()
words = np.load("words.npy", allow_pickle=True)
classes = np.load("classes.npy", allow_pickle=True)

def clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

def bag_of_words(sentence):
    tokens = clean_sentence(sentence)
    bow = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bow[i] = 1
    return np.array(bow)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25  # Confidence level
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return classes[results[0][0]] if results else "default"

def get_response(intent):
    for i in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
    return "I'm not sure how to respond to that."

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response_api():
    user_input = request.json.get("user_input", "")
    intent = predict_class(user_input)
    response = get_response(intent)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)