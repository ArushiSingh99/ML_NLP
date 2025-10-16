# Sentiment Analysis API

## Overview

This project implements a **Sentiment Analysis API** using Python and Flask. It allows users to analyze the sentiment of text data, either **single sentences** or **batches of sentences**.

The API includes text preprocessing with **TextCleaner** and a placeholder **SentimentAnalyzer**. Participants can extend the ML model, add training, evaluation, and advanced preprocessing features.

---

## Features

* Clean and preprocess text: remove URLs, mentions, hashtags, punctuation, stopwords, expand contractions.
* Predict sentiment for a single text input.
* Predict sentiment for multiple texts in batch.
* Health check endpoint to verify API status.
* Placeholder sentiment model that can be replaced with real ML/DL models.
* Modular and extendable architecture.

---

## Endpoints

| Endpoint         | Method | Description                          | Input Example                                   |
| ---------------- | ------ | ------------------------------------ | ----------------------------------------------- |
| `/`              | GET    | API information and available routes | None                                            |
| `/health`        | GET    | Check API health                     | None                                            |
| `/predict`       | POST   | Predict sentiment of a single text   | `{ "text": "I love this product!" }`            |
| `/predict_batch` | POST   | Predict sentiment for multiple texts | `{ "texts": ["I love this!", "This is bad!"] }` |

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd sentiment_api
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
python app.py
```

The API will start at `http://0.0.0.0:5000`.

---

## Usage

### Predict single text

```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
    "text": "I love this product!"
}
```

### Predict batch of texts

```bash
POST http://localhost:5000/predict_batch
Content-Type: application/json

{
    "texts": ["I love this!", "This is terrible!"]
}
```

---

## Project Structure

```
sentiment_api/
│
├── app.py                     # Main Flask API
├── models/
│   └── sentiment_analyzer.py  # Placeholder sentiment model
├── preprocessing/
│   └── text_cleaner.py        # Text cleaning utilities
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## Contributing

See `contributing.md` for guidelines on how to contribute to this project. Participants can extend the API by:

* Integrating real ML/DL models for sentiment prediction.
* Adding training and evaluation endpoints.
* Implementing authentication and rate-limiting.
* Enhancing text preprocessing (emoji handling, slang expansion, etc.).

---

## License

This project is part of the **Sentiment Analysis Project** submission. License details can be added here if needed.

