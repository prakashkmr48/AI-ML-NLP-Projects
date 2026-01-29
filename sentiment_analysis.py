"""Sentiment Analysis using TextBlob
This script demonstrates sentiment analysis on text data using NLP.
"""

from textblob import TextBlob
import pandas as pd
from typing import List, Tuple

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze sentiment of given text.
    
    Args:
        text: Input text string
    
    Returns:
        Tuple of (sentiment_label, polarity_score)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, polarity

def batch_analyze(texts: List[str]) -> pd.DataFrame:
    """
    Analyze sentiment for multiple texts.
    
    Args:
        texts: List of text strings
    
    Returns:
        DataFrame with text, sentiment, and polarity
    """
    results = []
    for text in texts:
        sentiment, polarity = analyze_sentiment(text)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'polarity': polarity
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible and disappointing.",
        "It's okay, nothing special.",
        "Great quality and fast delivery!",
        "Worst experience ever, very unhappy."
    ]
    
    print("Sentiment Analysis Results:\n")
    results_df = batch_analyze(sample_texts)
    print(results_df.to_string(index=False))
    
    # Single text analysis example
    text = "Artificial Intelligence is transforming the world!"
    sentiment, score = analyze_sentiment(text)
    print(f"\nSingle Analysis:")
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Polarity Score: {score:.4f}")
