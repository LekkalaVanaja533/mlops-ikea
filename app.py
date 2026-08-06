from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline(
    "sentiment-analysis",
    model="./model"
)

@app.route("/")
def home():
    return "Sentiment Analysis API Running"

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    result = classifier(text)
    return jsonify(result)

app.run(host="0.0.0.0", port=5000)