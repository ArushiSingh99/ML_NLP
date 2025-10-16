from flask import Flask, request, jsonify
from models.sentiment_analyzer import SentimentAnalyzer
from preprocessing.text_cleaner import TextCleaner

app = Flask(__name__)

# Initialize
analyzer = SentimentAnalyzer()
text_cleaner = TextCleaner()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "health": "/health"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_trained": True,
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text field is required"}), 400
    cleaned_text = text_cleaner.clean_text(text)
    result = analyzer.predict(cleaned_text)
    return jsonify({"original_text": text, "cleaned_text": cleaned_text, **result})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    texts = data.get('texts', [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Texts field must be a list"}), 400
    cleaned_texts = text_cleaner.clean_texts(texts)
    results = analyzer.predict_batch(cleaned_texts)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
