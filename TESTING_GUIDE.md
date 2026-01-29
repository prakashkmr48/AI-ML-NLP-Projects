# Testing Guide for AI-ML-NLP-Projects

This guide will help you set up and test the AI/ML projects in this repository.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (to clone the repository)

## Step 1: Clone the Repository

```bash
git clone https://github.com/prakashkmr48/AI-ML-NLP-Projects.git
cd AI-ML-NLP-Projects
```

## Step 2: Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

Install the required packages:

```bash
pip install textblob scikit-learn pandas numpy

# Download TextBlob corpora (required for sentiment analysis)
python -m textblob.download_corpora
```

## Testing Individual Scripts

### 1. Test Sentiment Analysis

```bash
python sentiment_analysis.py
```

**Expected Output:**
- Sentiment analysis results for sample texts
- Sentiment labels (Positive/Negative/Neutral)
- Polarity scores for each text
- Single text analysis example

**Sample Output:**
```
Sentiment Analysis Results:

                                                    text sentiment  polarity
       I absolutely love this product! It's amazing!  Positive  0.625000
                     This is terrible and disappointing.  Negative -0.666667
                              It's okay, nothing special.   Neutral  0.000000
```

### 2. Test Text Classification

```bash
python text_classification_ml.py
```

**Expected Output:**
- Model training progress
- Accuracy score (typically 50-80% on the small sample dataset)
- Classification report with precision, recall, F1-score
- Sample predictions for new texts

**Sample Output:**
```
Training Text Classification Model...

Model Accuracy: 66.67%

Classification Report:
              precision    recall  f1-score   support

    Business       0.67      1.00      0.80         2
      Sports       1.00      0.50      0.67         2

Sample Predictions:
Text: New smartphone released with advanced AI features
Predicted Category: Technology
```

## Testing with Your Own Data

### Sentiment Analysis - Custom Text

```python
from sentiment_analysis import analyze_sentiment

text = "Your custom text here"
sentiment, polarity = analyze_sentiment(text)
print(f"Sentiment: {sentiment}, Polarity: {polarity}")
```

### Text Classification - Custom Training

```python
from text_classification_ml import TextClassifier

# Your custom texts and labels
texts = ["text1", "text2", "text3"]
labels = ["category1", "category2", "category1"]

classifier = TextClassifier()
accuracy, report = classifier.train(texts, labels)
print(f"Accuracy: {accuracy}")

# Predict new texts
predictions = classifier.predict(["new text to classify"])
print(predictions)
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError
**Error:** `ModuleNotFoundError: No module named 'textblob'`

**Solution:**
```bash
pip install textblob scikit-learn pandas numpy
```

### Issue 2: TextBlob Corpora Missing
**Error:** `LookupError: Resource punkt not found`

**Solution:**
```bash
python -m textblob.download_corpora
```

### Issue 3: Low Accuracy in Text Classification
**Reason:** The sample dataset is very small (15 examples)

**Solution:** Use a larger dataset with more training examples for better accuracy

## Running Tests in Jupyter Notebook

You can also test these scripts in Jupyter Notebook:

```bash
pip install jupyter
jupyter notebook
```

Create a new notebook and copy-paste the code from the scripts, running cells individually.

## Performance Benchmarks

| Script | Execution Time | Memory Usage |
|--------|---------------|-------------|
| sentiment_analysis.py | ~2-3 seconds | ~50 MB |
| text_classification_ml.py | ~3-5 seconds | ~100 MB |

## Next Steps

1. **Expand the dataset** - Add more training examples for better accuracy
2. **Try different models** - Experiment with SVM, Random Forest, or Neural Networks
3. **Feature engineering** - Add more text features like n-grams, word embeddings
4. **Hyperparameter tuning** - Optimize model parameters for better performance

## Need Help?

If you encounter any issues:
1. Check the error message carefully
2. Ensure all dependencies are installed
3. Verify Python version compatibility
4. Open an issue on GitHub for support

## Additional Resources

- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NLP with Python Tutorial](https://www.nltk.org/book/)

---

**Happy Testing! 🚀**
