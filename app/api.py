from flask import Flask, request, jsonify
from src.inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # TODO: Get text from request
    # TODO: Validate input
    # TODO: Call predict() function
    # TODO: Return JSON response with sentiment and confidence
    pass

@app.route('/health', methods=['GET'])
def health():
    # TODO: Return model status and version
    pass