from textblob import TextBlob
import pandas as pd

class SentimentModel:

    def analyze_text(self, text):
        if pd.isna(text):
            return "Neutral"

        score = TextBlob(str(text)).sentiment.polarity

        if score < -0.2:
            return "Negative"
        elif score > 0.2:
            return "Positive"
        return "Neutral"

    def apply_model(self, df, review_col="Reviews"):
        df["Polarity"] = df[review_col].apply(
            lambda t: TextBlob(str(t)).sentiment.polarity
        )
        df["Sentiment"] = df[review_col].apply(self.analyze_text)
        return df
