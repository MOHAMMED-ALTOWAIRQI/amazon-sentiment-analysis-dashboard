class AIRecommender:
    def generate(self, df):
        pos = len(df[df["Sentiment"] == "Positive"])
        neg = len(df[df["Sentiment"] == "Negative"])
        total = len(df)

        insights = []

        if total == 0:
            return ["No reviews available to generate insights."]

        if neg / total > 0.30:
            insights.append("A high percentage of negative reviews indicates potential product issues or customer dissatisfaction.")

        if pos / total > 0.60:
            insights.append("The product has strong positive sentiment. Consider increasing marketing efforts to boost sales.")

        if df["Polarity"].mean() < 0:
            insights.append("Average sentiment is negative. Customers are reporting recurring issues.")

        if len(insights) == 0:
            insights.append("Overall sentiment is good and no major issues detected.")

        return insights
