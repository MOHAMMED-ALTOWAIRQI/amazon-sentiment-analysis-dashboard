import pandas as pd

class ProductAnalyzer:

    def filter_product(self, df, keyword):
        if "Product Name" not in df.columns:
            print("ERROR: 'Product Name' column not found!")
            return pd.DataFrame()   

        keyword = str(keyword).strip().lower()
        df["Product Name"] = df["Product Name"].astype(str).str.lower()

        filtered = df[df["Product Name"].str.contains(keyword, na=False)]
        return filtered

    def get_review_stats(self, df):
        if df.empty:
            return None

        lengths = df["Reviews"].astype(str).apply(len)
        return {
            "count": len(df),
            "columns": df.shape[1],
            "min_length": lengths.min(),
            "max_length": lengths.max(),
            "avg_length": lengths.mean()
        }

    def top_products(self, df, n=10):
        if df.empty:
            return None
        return df["Product Name"].value_counts().head(n)
