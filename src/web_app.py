from flask import Flask, request, jsonify
from predict import predict_text

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    prediction = predict_text(text)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
