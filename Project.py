import numpy as np
import pywt
from scipy import signal
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class EOGClassifier:
    def __init__(self, sampling_rate=256):
        self.data = None
        self.labels = None
        self.features = None
        self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.sampling_rate = sampling_rate
        self.wavelet_families = ['db1', 'db2', 'db3', 'db4']
        self.is_trained = False

    def preprocess_signal(self, data):
        """Preprocess the raw signals by applying mean removal, bandpass filtering, and normalization."""
        # DC removal
        data = data - np.mean(data, axis=1, keepdims=True)

        # Bandpass filter (0.5-20 Hz)
        nyquist = self.sampling_rate / 2
        low, high = 0.5 / nyquist, 20 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 1, data)

        # Normalize
        normalized_data = (filtered_data - filtered_data.mean(axis=1, keepdims=True)) / (filtered_data.std(axis=1, keepdims=True) + 1e-8)

        return normalized_data

    def extract_features(self, data):
        """Extract wavelet-based features from the signals."""
        features = []
        for signal in data:
            signal_features = []
            for wavelet in self.wavelet_families:
                coeffs = pywt.wavedec(signal, wavelet, level=2)
                for coeff in coeffs:
                    signal_features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
            features.append(signal_features)
        return np.array(features)

    def load_data(self, filename):
        """Load data from a file."""
        signal_data = []
        try:
            with open(filename, 'r') as file:
                for line in file:
                    values = [float(value) for value in line.strip().split()]
                    signal_data.append(values)
            return np.array(signal_data)
        except Exception as e:
            raise ValueError(f"Error loading data from {filename}: {e}")

    def set_labels(self, label, num_samples):
        """Set the labels for the dataset."""
        labels = [label] * num_samples
        self.labels = self.label_encoder.fit_transform(labels)

    def train_model(self, data, labels):
        """Train the KNN classification model."""
        if len(data) != len(labels):
            raise ValueError("The number of samples and labels must be the same.")

        # Preprocess and extract features
        preprocessed_data = self.preprocess_signal(data)
        features = self.extract_features(preprocessed_data)

        if not self.scaler or not self.label_encoder or not self.model:
            raise ValueError("Scaler, LabelEncoder, or Model is not initialized.")

        # Fit the scaler and transform features
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        # Fit the label encoder with all unique labels
        self.label_encoder.fit(np.unique(labels))

        # Encode labels
        encoded_labels = self.label_encoder.transform(labels)

        # Train the KNN model
        self.model.fit(scaled_features, encoded_labels)
        self.is_trained = True

        # Evaluate accuracy
        accuracy = self.model.score(scaled_features, encoded_labels)

        return accuracy, preprocessed_data

    def predict(self, test_data):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")

        if not self.scaler or not self.label_encoder or not self.model:
            raise ValueError("Scaler, LabelEncoder, or Model is not initialized.")

        # Preprocess and extract features from test data
        preprocessed_data = self.preprocess_signal(test_data)
        features = self.extract_features(preprocessed_data)
        scaled_features = self.scaler.transform(features)

        try:
            # Predict using the trained KNN model
            predictions = self.model.predict(scaled_features)

            # Decode labels
            if hasattr(self.label_encoder, "classes_"):
                decoded_predictions = self.label_encoder.inverse_transform(predictions)
            else:
                raise ValueError("LabelEncoder is not fitted properly.")

            return decoded_predictions
        except Exception as e:
            raise ValueError(f"Prediction error: {e}")

    def evaluate_model(self, test_data, true_labels):
        """Evaluate the model on test data."""
        predictions = self.predict(test_data)
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, target_names=np.unique(true_labels).astype(str))
        cm = confusion_matrix(true_labels, predictions)

        print("Test Accuracy: {:.2f}".format(accuracy))
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)

        return accuracy, report, cm

