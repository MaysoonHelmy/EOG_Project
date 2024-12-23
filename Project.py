import numpy as np
import pywt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class EOGClassifier:
    def __init__(self, sampling_rate=256):
        self.data = None
        self.labels = None
        self.features = None
        self.model = None
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
        normalized_data = (filtered_data - filtered_data.mean(axis=1, keepdims=True)) / filtered_data.std(axis=1, keepdims=True)

        return normalized_data

    def extract_features(self, data):
        """Extract wavelet-based features from the signals."""
        features = []
        for signal in data:
            signal_features = []
            for wavelet in self.wavelet_families:
                coeffs = pywt.wavedec(signal, wavelet, level=2)
                for coeff in coeffs:
                    signal_features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        np.max(coeff),
                        np.min(coeff)
                    ])
            features.append(signal_features)
        return np.array(features)

    def load_data(self, filename):
        """Load data from a file."""
        signal = []
        try:
            with open(filename, 'r') as file:
                for line in file:
                    values = [float(value) for value in line.strip().split()]
                    signal.append(values)
            return np.array(signal)
        except Exception as e:
            raise ValueError(f"Error loading data from {filename}: {e}")

    def set_labels(self, label, num_samples):
        """Set the labels for the dataset."""
        labels = [label] * num_samples
        self.labels = self.label_encoder.fit_transform(labels)

    def train_model(self):
        """Train the K-Nearest Neighbors classifier."""
        if self.data is None or self.labels is None:
            raise ValueError("Data and labels must be set before training")

        # Preprocess the data
        preprocessed_data = self.preprocess_signal(self.data)
        features = self.extract_features(preprocessed_data)
        self.features = self.scaler.fit_transform(features)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )

        # Train the model
        self.model = KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='minkowski', p=2
        )
        self.model.fit(X_train, y_train)

        # Evaluate the model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Training complete.")
        print(f"Accuracy on test data: {accuracy:.2f}")
        self.is_trained = True

        return accuracy

    def predict(self, test_data):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        preprocessed_data = self.preprocess_signal(test_data)
        features = self.extract_features(preprocessed_data)
        scaled_features = self.scaler.transform(features)
        predictions = self.model.predict(scaled_features)

        return self.label_encoder.inverse_transform(predictions)

    def evaluate_model(self, test_data, true_labels):
        """Evaluate the model on test data."""
        predictions = self.predict(test_data)
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        return predictions, accuracy, report

