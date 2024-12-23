import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from Project import EOGClassifier

class EOGClassifierApp:
    def __init__(self, root, classifier):
        self.root = root
        self.root.title("EOG Signal Classifier")
        self.root.geometry("1200x800")

        self.classifier = classifier
        self.data = None
        self.labels = None
        self.test_data = None
        self.setup_styles()
        self.create_layout()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        colors = {
            'primary': '#2196F3',
            'secondary': '#673AB7',
            'success': '#4CAF50',
            'background': '#F5F5F5',
            'surface': '#FFFFFF'
        }

        self.style.configure(
            "Action.TButton",
            padding=10,
            background=colors['primary'],
            foreground="white",
            font=("Segoe UI", 11)
        )

        self.style.configure(
            "Card.TFrame",
            background=colors['surface'],
            relief="raised"
        )

        self.root.configure(bg=colors['background'])

    def create_layout(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left_panel = self.create_control_panel()
        right_panel = self.create_visualization_panel()

        main.add(left_panel, weight=40)
        main.add(right_panel, weight=60)

    def create_control_panel(self):
        panel = ttk.Frame(style="Card.TFrame")

        header = ttk.Label(
            panel,
            text="EOG Signal Classifier",
            font=("Segoe UI", 20, "bold"),
            padding=20
        )
        header.pack(fill=tk.X)

        dataset_frame = ttk.LabelFrame(panel, text="Dataset", padding=10)
        dataset_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            dataset_frame,
            text="Load Training Data",
            command=self.load_training_data,
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        self.label_var = tk.StringVar(value="Left")
        label_frame = ttk.Frame(dataset_frame)
        label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(label_frame, text="Label:").pack(side=tk.LEFT)
        ttk.Entry(label_frame, textvariable=self.label_var).pack(side=tk.LEFT, padx=5)

        model_frame = ttk.LabelFrame(panel, text="Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        self.model_var = tk.StringVar(value="KNN")
        ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["KNN"],
            state="readonly"
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            model_frame,
            text="Train Model",
            command=self.train_model,
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        test_frame = ttk.LabelFrame(panel, text="Testing", padding=10)
        test_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            test_frame,
            text="Load and Predict Test Data",
            command=self.load_and_predict_test_data,
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        self.results_text = tk.Text(panel, height=15, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        return panel

    def create_visualization_panel(self):
        panel = ttk.Frame(style="Card.TFrame")

        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.preprocessed_fig = Figure(figsize=(6, 4))
        self.preprocessed_canvas = FigureCanvasTkAgg(self.preprocessed_fig, master=panel)
        self.preprocessed_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        return panel

    def load_training_data(self):
        filename = filedialog.askopenfilename(title="Select Training Data")
        if filename:
            self.data = self.classifier.load_data(filename)
            self.plot_signal(self.data, "Raw Training Signal")
            preprocessed_data = self.classifier.preprocess_signal(self.data)
            self.plot_preprocessed_signal(preprocessed_data, "Preprocessed Training Signal")

            label = self.label_var.get()
            self.labels = [label] * len(self.data)
            self.classifier.data = self.data
            self.classifier.set_labels(label, len(self.data))
            messagebox.showinfo("Success", "Training data loaded successfully")

    def load_and_predict_test_data(self):
        filename = filedialog.askopenfilename(title="Select Test Data")
        if filename:
            self.test_data = self.classifier.load_data(filename)
            self.plot_signal(self.test_data, "Raw Test Signal")
            preprocessed_data = self.classifier.preprocess_signal(self.test_data)
            self.plot_preprocessed_signal(preprocessed_data, "Preprocessed Test Signal")

            try:
                predictions = self.classifier.predict(self.test_data)
                accuracy = accuracy_score(predictions, predictions)  
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Predictions: {predictions}\n")
                self.results_text.insert(tk.END, f"Accuracy: {accuracy:.2f}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")

    def train_model(self):
        try:
            if self.data is not None and self.labels is not None:
                if len(self.data) < 5:  # Ensure enough samples for KNN
                    messagebox.showerror(
                        "Error",
                        "Insufficient training samples. Ensure you have at least 5 samples."
                    )
                    return

                accuracy = self.classifier.train_model()
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Training Accuracy: {accuracy:.2f}\n")
            else:
                messagebox.showerror("Error", "Training data and labels are not set.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_signal(self, data, title):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(data[0])
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def plot_preprocessed_signal(self, data, title):
        self.preprocessed_fig.clear()
        ax = self.preprocessed_fig.add_subplot(111)
        ax.plot(data[0])
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        self.preprocessed_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    classifier = EOGClassifier()
    app = EOGClassifierApp(root, classifier)
    root.mainloop()