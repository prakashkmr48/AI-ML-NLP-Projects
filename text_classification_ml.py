"""Text Classification using Machine Learning
Demonstrates text classification with scikit-learn and TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from typing import Tuple, List

class TextClassifier:
    """Text classification model using TF-IDF and Naive Bayes."""
    
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def train(self, texts: List[str], labels: List[str]) -> Tuple[float, str]:
        """
        Train the text classifier.
        
        Args:
            texts: List of text documents
            labels: List of corresponding labels
        
        Returns:
            Tuple of (accuracy_score, classification_report)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize text data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_tfidf, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for new texts.
        
        Args:
            texts: List of text documents to classify
        
        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict(X_tfidf)

if __name__ == "__main__":
    # Sample dataset: News category classification
    texts = [
        "The stock market rallied today with tech stocks leading gains",
        "Scientists discover new method for treating cancer",
        "Championship game ends in dramatic overtime victory",
        "Government announces new economic stimulus package",
        "Breakthrough in AI research enables faster training",
        "Team wins finals after incredible comeback",
        "New medical device approved by regulatory agency",
        "Federal Reserve raises interest rates again",
        "Athletes prepare for upcoming international competition",
        "Researchers publish findings on climate change impact",
        "Market indices close at record highs",
        "Hospital implements innovative treatment protocol",
        "Olympic gold medalist announces retirement",
        "Central bank adjusts monetary policy",
        "Study reveals new insights into human genome"
    ]
    
    labels = [
        "Business", "Health", "Sports", "Business", "Technology",
        "Sports", "Health", "Business", "Sports", "Science",
        "Business", "Health", "Sports", "Business", "Science"
    ]
    
    # Train classifier
    print("Training Text Classification Model...\n")
    classifier = TextClassifier(max_features=1000)
    accuracy, report = classifier.train(texts, labels)
    
    print(f"Model Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(report)
    
    # Test predictions
    print("\nSample Predictions:")
    test_texts = [
        "New smartphone released with advanced AI features",
        "Doctor recommends preventive health measures",
        "Team scores winning goal in final minutes"
    ]
    
    predictions = classifier.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted Category: {pred}\n")
