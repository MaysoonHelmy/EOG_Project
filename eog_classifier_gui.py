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
        self.test_label = None
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
            text="Load Training Data (Right)",
            command=lambda: self.load_training_data("Right"),
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            dataset_frame,
            text="Load Training Data (Left)",
            command=lambda: self.load_training_data("Left"),
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        model_frame = ttk.LabelFrame(panel, text="Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

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
            text="Load Test Data",
            command=self.load_test_data,
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            test_frame,
            text="Predict",
            command=self.predict,
            style="Action.TButton"
        ).pack(fill=tk.X, pady=2)

        # Scrollable text log
        log_frame = ttk.Frame(panel)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(log_frame, height=15, width=60, wrap=tk.WORD)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

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

    def load_training_data(self, label):
        filename = filedialog.askopenfilename(title=f"Select Training Data ({label})")
        if filename:
            data = self.classifier.load_data(filename)
            num_samples = len(data)
            self.classifier.set_labels(label, num_samples)
            if self.data is None:
                self.data = data
                self.labels = np.array([label] * len(data))
            else:
                self.data = np.vstack((self.data, data))
                self.labels = np.concatenate((self.labels, np.array([label] * len(data))))
            self.plot_signal(data, f"Raw Training Signal ({label})")
            messagebox.showinfo("Success", f"Training data for {label} loaded successfully")

    def load_test_data(self):
        filename = filedialog.askopenfilename(title="Select Test Data")
        if filename:
            self.test_data = self.classifier.load_data(filename)
            self.plot_signal(self.test_data, "Raw Test Signal")
            messagebox.showinfo("Success", "Test data loaded successfully")

    def train_model(self):
        try:
            if self.data is not None and self.labels is not None:
                accuracy, preprocessed_data = self.classifier.train_model(self.data, self.labels)
                self.results_text.insert(tk.END, f"Model trained successfully!\n")
                self.results_text.insert(tk.END, f"Training Accuracy: {accuracy:.2f}\n")

                # Train Classification Report
                train_report = classification_report(self.labels, self.classifier.predict(self.data), target_names=np.unique(self.labels).astype(str))
                self.results_text.insert(tk.END, f"Training Classification Report:\n{train_report}\n")

                self.plot_signal(self.data, "Raw Training Signal")
                self.plot_preprocessed_signal(preprocessed_data, "Preprocessed Signal")
            else:
                messagebox.showerror("Error", "No training data loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict(self):
        try:
            if self.test_data is not None:
                # Make predictions
                predictions = self.classifier.predict(self.test_data)
                self.results_text.insert(tk.END, f"Predictions: {predictions}\n")

                # Calculate test accuracy and classification report
                if self.test_label is not None:
                    test_accuracy = accuracy_score(self.test_label, predictions)
                    self.results_text.insert(tk.END, f"Test Accuracy: {test_accuracy:.2f}\n")

                    # Test Classification Report
                    test_report = classification_report(self.test_label, predictions, target_names=np.unique(self.test_label).astype(str))
                    self.results_text.insert(tk.END, f"Test Classification Report:\n{test_report}\n")
                else:
                    self.results_text.insert(tk.END, "Test labels are not provided.\n")
            else:
                messagebox.showerror("Error", "Test data is not loaded.")
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
