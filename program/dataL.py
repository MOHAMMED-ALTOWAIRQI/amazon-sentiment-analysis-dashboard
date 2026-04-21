import pandas as pd
import zipfile

class DataLoader:

    def load_zip(self, path):
        with zipfile.ZipFile(path, 'r') as zf:
            csv_name = [f for f in zf.namelist() if f.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
        return df

    def load_csv(self, path):
        return pd.read_csv(path)

    def load_excel(self, path):
        return pd.read_excel(path)
