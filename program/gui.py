import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from dataL import DataLoader
from sentiment import SentimentModel
from analyzer import ProductAnalyzer
from ai_recommend import AIRecommender
from charts import ChartBuilder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from ML import MLModel
import pandas as pd
from tkinter import simpledialog

class SentimentAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analyzer Dashboard")
        self.root.geometry("1000x900")

        self.df = None
        self.product_df = None

        self.loader = DataLoader()
        self.model = SentimentModel()
        self.analyzer = ProductAnalyzer()
        self.recommender = AIRecommender()
        self.charts = ChartBuilder()
        self.ml = MLModel()  

        title = tk.Label(root, text="Sentiment Analyzer Dashboard", font=("Arial", 22, "bold"))
        title.pack(pady=10)

        file_frame = tk.Frame(root)
        file_frame.pack(pady=10)

        tk.Button(
            file_frame,
            text="Load File",
            font=("Arial", 12),
            width=20,
            command=self.load_file
        ).grid(row=0, column=0, padx=10)

        self.file_type = ttk.Combobox(file_frame, values=["ZIP", "CSV", "Excel"], width=15)
        self.file_type.set("ZIP")
        self.file_type.grid(row=0, column=1, padx=10)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Select Product:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        self.input_product = ttk.Combobox(control_frame, width=50, font=("Arial", 12), state="readonly")
        self.input_product.grid(row=0, column=1, padx=5)
        self.input_product.set("Select Product")

        tk.Button(
            control_frame,
            text="Analyze",
            font=("Arial", 12),
            width=15,
            command=self.analyze_product
        ).grid(row=0, column=2, padx=5)

        tk.Label(control_frame, text="Chart Type:", font=("Arial", 12)).grid(row=1, column=0, padx=5)

        self.chart_type = ttk.Combobox(
            control_frame,
            values=["Bar", "Pie", "Histogram", "Scatter", "Line"],
            width=20,
            state="readonly"
        )
        self.chart_type.set("Bar")
        self.chart_type.grid(row=1, column=1, pady=5)

        tk.Button(
            control_frame,
            text="Show Chart",
            font=("Arial", 12),
            width=15,
            command=self.show_chart
        ).grid(row=1, column=2, padx=5)

        # TOP N
        tk.Label(control_frame, text="Top N Products:", font=("Arial", 12)).grid(row=2, column=0)
        self.top_n = tk.Entry(control_frame, width=10)
        self.top_n.grid(row=2, column=1, padx=5)
        self.top_n.insert(0, "10")

        tk.Button(
            control_frame,
            text="Show Top",
            font=("Arial", 12),
            width=15,
            command=self.show_top_products
        ).grid(row=2, column=2, padx=5)

        tk.Button(
            control_frame,
            text="AI Insights",
            font=("Arial", 12),
            width=15,
            command=self.show_ai_insights
        ).grid(row=3, column=1, pady=10)

        self.output = scrolledtext.ScrolledText(root, width=110, height=30, font=("Arial", 11))
        self.output.pack(pady=10)
        tk.Button(
            control_frame,
            text="Export",
            font=("Arial", 12),
            width=15,
            command=self.export_file
        ).grid(row=3, column=2, padx=5)

    def load_file(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        try:
            if self.file_type.get() == "ZIP":
                self.df = self.loader.load_zip(path)
            elif self.file_type.get() == "CSV":
                self.df = self.loader.load_csv(path)
            elif self.file_type.get() == "Excel":
                self.df = self.loader.load_excel(path)
            else:
                messagebox.showerror("Error", "Unsupported file type")
                return

            self.df.columns = [c.strip() for c in self.df.columns]

            if "Product Name" not in self.df.columns:
                messagebox.showerror("Error", "Column 'Product Name' not found")
                return

            products = sorted(self.df["Product Name"].dropna().unique())
            self.input_product["values"] = products
            self.input_product.set("Select Product")

            messagebox.showinfo("Success", "Dataset loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def analyze_product(self):
        if self.df is None:
            messagebox.showerror("Error", "Load a dataset first")
            return

        selected = self.input_product.get().strip()
        print("Selected product:", selected)

        self.product_df = self.df[self.df["Product Name"] == selected]

        if len(self.product_df) == 0:
            self.product_df = self.df[self.df["Product Name"].str.contains(selected, case=False, na=False)]

        if len(self.product_df) == 0:
            messagebox.showerror("Error", "No matching products found")
            return

        self.product_df = self.model.apply_model(self.product_df)

        acc = self.ml.train(self.product_df)
        self.output.delete("1.0", tk.END)

        self.output.insert(tk.END, "\n--- Machine Learning Model ---\n")
        self.output.insert(tk.END, f"ML Model: Logistic Regression\n")
        self.output.insert(tk.END, f"Training Samples: {len(self.product_df)}\n")
        self.output.insert(tk.END, f"Training Accuracy: {acc*100:.2f}%\n\n")


        stats = self.analyzer.get_review_stats(self.product_df)

        self.output.insert(tk.END, f"Product: {selected}\n")
        self.output.insert(tk.END, f"Count: {stats['count']}\n")
        self.output.insert(tk.END, f"Columns: {stats['columns']}\n")
        self.output.insert(tk.END, f"Shortest: {stats['min_length']}\n")
        self.output.insert(tk.END, f"Longest: {stats['max_length']}\n")
        self.output.insert(tk.END, f"Average Length: {stats['avg_length']:.2f}\n\n")

        for s in ["Positive", "Neutral", "Negative"]:
            review = self.product_df[self.product_df["Sentiment"] == s].head(1)
            self.output.insert(tk.END, f"{s}:\n")
            self.output.insert(tk.END, str(review[["Reviews", "Polarity"]]) + "\n\n")

    def show_chart(self):
        if self.product_df is None:
            messagebox.showerror("Error", "Analyze a product first")
            return

        chart = self.chart_type.get()

        try:
            if chart == "Bar":
                fig = self.charts.bar_chart(self.product_df)
            elif chart == "Pie":
                fig = self.charts.pie_chart(self.product_df)
            elif chart == "Histogram":
                fig = self.charts.histogram(self.product_df)
            elif chart == "Line":
                fig = self.charts.line_chart(self.product_df)
            elif chart == "Scatter":
                fig = self.charts.scatter_plot(self.product_df)
            elif chart == "Heatmap":
                fig = self.charts.heatmap(self.product_df)
            else:
                messagebox.showerror("Error", "Invalid chart type")
                return

            win = tk.Toplevel(self.root)
            win.title(f"{chart} Chart")

            canvas = FigureCanvasTkAgg(fig, win)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Chart Error", str(e))

    def show_top_products(self):
        if self.df is None:
            messagebox.showerror("Error", "Load dataset first")
            return

        try:
            n = int(self.top_n.get())
        except:
            messagebox.showerror("Error", "Invalid number")
            return

        top = self.analyzer.top_products(self.df, n)

        win = tk.Toplevel(self.root)
        win.title("Top Products")

        text = tk.Text(win, width=60, height=30, font=("Arial", 11))
        text.pack()

        text.insert(tk.END, str(top))

    def show_ai_insights(self):
        if self.product_df is None:
            messagebox.showerror("Error", "Analyze a product first")
            return

        insights = self.recommender.generate(self.product_df)

        self.output.insert(tk.END, "\nAI Insights:\n")
        for i in insights:
            self.output.insert(tk.END, f"- {i}\n")
    def export_file(self):
        import pandas as pd

        if self.product_df is None and self.df is None:
            messagebox.showerror("Error", "No data to export. Load and analyze a product first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Choose Export Options")
        win.geometry("400x250")

        tk.Label(win, text="Select Data to Export:", font=("Arial", 12)).pack(pady=10)

        choice_var = tk.StringVar()
        choice_var.set("Raw Data")

        options = ["Raw Data", "Product Data", "Sentiment Results"]

        menu = ttk.Combobox(win, textvariable=choice_var, values=options, state="readonly")
        menu.pack(pady=5)

    def export_file(self):

        if self.product_df is None and self.df is None:
            messagebox.showerror("Error", "No data to export. Load and analyze a product first.")
            return

        win = tk.Toplevel(self.root) 
        win.title("Choose Export Options")
        win.geometry("400x250")

        tk.Label(win, text="Select Data to Export:", font=("Arial", 12)).pack(pady=10)

        choice_var = tk.StringVar() 
        choice_var.set("Raw Data")

        options = ["Raw Data", "Product Data", "Sentiment Results"]

        menu = ttk.Combobox(win, textvariable=choice_var, values=options, state="readonly")
        menu.pack(pady=5)

     
        def finalize_export(): 
            choice = choice_var.get()

            df = None

            if choice == "Raw Data":
                df = self.df
            elif choice == "Product Data":
                df = self.product_df
            elif choice == "Sentiment Results":
                if self.product_df is None:
                    messagebox.showerror("Error", "Product Data not analyzed yet!")
                    return
                df = self.product_df[["Product Name", "Reviews", "Sentiment", "Polarity"]]
            else:
                messagebox.showerror("Error", "Invalid export option selected")
                return

            if df is None or len(df) == 0:
                 messagebox.showerror("Error", "Selected data is empty.")
                 return

            file_path = filedialog.asksaveasfilename(
                title="Export File",
                defaultextension=".*",
                filetypes=[
                    ("CSV File", "*.csv"),
                    ("Excel File", "*.xlsx"),
                    ("Text File", "*.txt"),
                    ("JSON File", "*.json")
                ]
            )

            if not file_path:
                return

            try:
                if file_path.endswith(".csv"):
                    df.to_csv(file_path, index=False)
                elif file_path.endswith(".xlsx"):
                    df.to_excel(file_path, index=False)
                elif file_path.endswith(".json"):
                    df.to_json(file_path, orient="records")
                elif file_path.endswith(".txt"):
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(df.to_string())
                else:
                    messagebox.showerror("Error", "Unsupported file type")
                    return

                messagebox.showinfo("Success", "File exported successfully!")
                win.destroy() 

            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{e}")

        tk.Button(
            win,
            text="Export Now",
            font=("Arial", 12),
            width=15,
            command=finalize_export
        ).pack(pady=20)        


