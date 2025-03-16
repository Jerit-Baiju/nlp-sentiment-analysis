import os
import re
import sys

import joblib
import kagglehub
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

nltk.download("punkt_tab")

# Step 1: Download the dataset
dataset_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

# Step 2: Locate the CSV file
csv_file = [f for f in os.listdir(dataset_path) if f.endswith(".csv")][0]
csv_path = os.path.join(dataset_path, csv_file)

# Step 3: Load the dataset
df = pd.read_csv(csv_path)

# Step 4: Download stopwords & tokenizer (only runs once)
nltk.download("stopwords")
nltk.download("punkt")

# Step 5: Load stopwords
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Cleans the given text by removing HTML tags, punctuation, and stopwords."""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 3. Remove punctuation & special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 4. Tokenize (split text into words)
    words = word_tokenize(text)

    # 5. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # 6. Join words back into a single string
    return " ".join(words)


# Step 6: Apply the cleaning function to the dataset
df["cleaned_review"] = df["review"].apply(clean_text)

# Step 7: Show the first few cleaned reviews
print(df[["review", "cleaned_review"]].head())

# Step 8: Print dataset info
print("\nDataset Info:")
print(df.info())

# Step 9: Prepare data for sentiment analysis model
# Map sentiment labels to numeric values
df["sentiment_numeric"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_review"], df["sentiment_numeric"], test_size=0.2, random_state=42
)

# Step 10: Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 11: Build and train the sentiment analysis model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 12: Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Step 13: Save the model and vectorizer for future use
model_dir = "/workspaces/nlp-sentiment-analysis/models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))


# Step 14: Function to analyze sentiment of new sentences
def analyze_sentiment(sentence):
    """
    Analyzes the sentiment of a given sentence using the trained model.
    Returns: tuple (prediction label, probability)
    """
    # Clean the input text
    cleaned_sentence = clean_text(sentence)

    # Transform the text using the same vectorizer
    sentence_tfidf = vectorizer.transform([cleaned_sentence])

    # Predict sentiment
    prediction = model.predict(sentence_tfidf)[0]

    # Get probability scores
    proba = model.predict_proba(sentence_tfidf)[0]
    confidence = proba[prediction]

    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence


# Example usage
if __name__ == "__main__":
    # Test with some example sentences
    test_sentences = [
        "This movie was fantastic, I really enjoyed it!",
        "The acting was terrible and the plot made no sense.",
        "It was ok, not great but not awful either.",
    ]

    print("\nSentiment Analysis Examples:")
    for sentence in test_sentences:
        sentiment, confidence = analyze_sentiment(sentence)
        print(f"Sentence: '{sentence}'")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})\n")
