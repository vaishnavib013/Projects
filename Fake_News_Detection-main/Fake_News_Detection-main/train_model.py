import pandas as pd
import numpy as np
import nltk
import pickle
import re
import string
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing setup
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = tokenizer.tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    return ""

def load_dataset_from_csv(true_csv_path, false_csv_path):
    true_df = pd.read_csv(true_csv_path)
    false_df = pd.read_csv(false_csv_path)

    true_df['label'] = 1  # Real
    false_df['label'] = 0  # Fake

    df = pd.concat([true_df, false_df], ignore_index=True)

    # Use 'text' column if it exists, otherwise 'title'
    if 'text' in df.columns:
        df['content'] = df['text']
    elif 'title' in df.columns:
        df['content'] = df['title']
    else:
        raise ValueError("CSV files must contain either a 'text' or 'title' column")

    df = df[['content', 'label']]
    return df

def train_and_save_model(csv_folder_path):
    true_csv = os.path.join(csv_folder_path, 'True.csv')
    false_csv = os.path.join(csv_folder_path, 'Fake.csv')

    print("Loading CSV datasets...")
    df = load_dataset_from_csv(true_csv, false_csv)
    print(f"Loaded {len(df)} records.")

    print("Preprocessing...")
    df['content'] = df['content'].apply(preprocess_text)

    X = df['content']
    y = df['label']

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training...")
    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("Saving model and vectorizer...")
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Done. Files saved in 'model/' folder.")

if __name__ == "__main__":
    dataset_path = "./News_dataset"

    train_and_save_model(dataset_path)
