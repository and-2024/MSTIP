from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Load trained model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalpha()]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't understand that. Can you rephrase?"

    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure how to respond to that."

def chatbot_response(msg):
    ints = predict_class(msg)
    return get_response(ints, intents)

@app.route("/chat", methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    response = chatbot_response(message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
