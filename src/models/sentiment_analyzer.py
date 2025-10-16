from typing import Dict, List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np

class SentimentAnalyzer:
    """Basic sentiment analyzer - participants need to implement ML models."""
    
    def __init__(self) -> None:
        """Initialize the sentiment analyzer."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        self.pipeline = make_pipeline(self.vectorizer, self.model)

        pass
        
    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the sentiment analysis model.
        
        Args:
            texts: List of training texts
            labels: List of corresponding sentiment labels
            
        TODO for participants:
        - Implement model training logic
        - Add data preprocessing
        - Choose and train ML model (Naive Bayes, SVM, Neural Network, etc.)
        - Store trained model for predictions
        """
        if not texts or not labels:
            raise ValueError("Training data or labels cannot be empty.")
        
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        
        # Optional: compute training accuracy
        preds = self.pipeline.predict(texts)
        self.accuracy = accuracy_score(labels, preds)
        pass
        
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with 'sentiment' and 'confidence' keys
            
        TODO for participants:
        - Implement prediction logic
        - Return sentiment (positive/negative/neutral)
        - Return confidence score (0.0 to 1.0)
        - Handle edge cases (empty text, etc.)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Please call train() first.")
        
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0}
        
        pred = self.pipeline.predict([text])[0]
        probs = self.pipeline.predict_proba([text])[0]
        confidence = float(np.max(probs))
        
        return {"sentiment": pred, "confidence": round(confidence, 3)}
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment predictions
            
        TODO for participants:
        - Implement batch prediction for efficiency
        - Process multiple texts at once
        - Return list of prediction dictionaries
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Please call train() first.")
        
        if not texts:
            return []
        
        preds = self.pipeline.predict(texts)
        probs = self.pipeline.predict_proba(texts)
        
        results = []
        for sentiment, prob in zip(preds, probs):
            confidence = float(np.max(prob))
            results.append({
                "sentiment": sentiment,
                "confidence": round(confidence, 3)
            })
        return results
        
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
            
        TODO for participants:
        - Return model type, training status, accuracy, etc.
        - Add model metadata
        """
        return{
            "model_type": "Multinomial Naive Bayes + TF-IDF",
            "status": "trained" if self.is_trained else "untrained",
            "training_accuracy": f"{self.accuracy:.2f}" if self.accuracy else "N/A"
        }


# TODO for participants - Additional features to implement:
# - Model evaluation and metrics
# - Hyperparameter tuning
# - Cross-validation
# - Model persistence (save/load)
# - Different ML algorithms (SVM, Random Forest, Neural Networks)
# - Feature engineering
# - Model comparison tools
# - Real-time prediction API
# - Batch processing optimization
