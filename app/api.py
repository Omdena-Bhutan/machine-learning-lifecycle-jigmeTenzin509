from flask import Flask, request, jsonify
from src.inference import predict

app = Flask(__name__)

from flask import Flask, request, jsonify
from src.inference import predict

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text field missing"}), 400

    result = predict(data["text"])
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "distilbert sentiment"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
