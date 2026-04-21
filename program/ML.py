import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib

class MLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.model = LogisticRegression(max_iter=200)

    def train(self, df, text_col="Reviews", label_col="Sentiment"):
        df = df[[text_col, label_col]].dropna()

        X = df[text_col].astype(str)
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)

        preds = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)

        return acc

    def predict(self, text):
        X = self.vectorizer.transform([str(text)])
        return self.model.predict(X)[0]

    def save(self, path="sentiment_model.pkl"):
        joblib.dump((self.model, self.vectorizer), path)

    def load(self, path="sentiment_model.pkl"):
        self.model, self.vectorizer = joblib.load(path)
