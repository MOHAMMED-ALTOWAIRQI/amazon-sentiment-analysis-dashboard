import matplotlib.pyplot as plt
import seaborn as sns

class ChartBuilder:

    def bar_chart(self, df):
        fig, ax = plt.subplots(figsize=(7, 4))
        sentiment_counts = df["Sentiment"].value_counts()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "blue", "red"])
        ax.set_title("Sentiment Distribution (Bar)")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        return fig

    def pie_chart(self, df):
        fig, ax = plt.subplots(figsize=(6, 6))
        sentiment_counts = df["Sentiment"].value_counts()
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax.set_title("Sentiment Distribution (Pie)")
        return fig

    def histogram(self, df):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["Polarity"], bins=30, color="purple", alpha=0.7)
        ax.set_title("Polarity Histogram")
        ax.set_xlabel("Polarity")
        ax.set_ylabel("Frequency")
        return fig

    def line_chart(self, df):
        df2 = df.reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df2.index, df2["Polarity"], color="blue")
        ax.set_title("Sentiment Polarity Over Index (Line Chart)")
        ax.set_xlabel("Review Index")
        ax.set_ylabel("Polarity")
        return fig

    def box_plot(self, df):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(df["Polarity"])
        ax.set_title("Polarity Distribution (Box Plot)")
        return fig

    def scatter_plot(self, df):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["Rating"], df["Polarity"], alpha=0.5, c="red")
        ax.set_title("Rating vs Polarity (Scatter Plot)")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Polarity")
        return fig




    