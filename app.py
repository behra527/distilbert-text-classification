from flask import Flask, request, jsonify, send_file
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import pickle

app = Flask(__name__)

# Load model, tokenizer, and label encoder
model = DistilBertForSequenceClassification.from_pretrained("./distilbert_recommender")
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_recommender")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_user_input(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    result = predict_user_input(user_input)
    return jsonify({"recommended_role": result})

if __name__ == "__main__":
    app.run(debug=True) 