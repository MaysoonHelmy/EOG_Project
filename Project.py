import numpy as np
import pywt
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class EOGClassifier:
    def __init__(self, sampling_rate=176):
        self.data = None
        self.labels = None
        self.features = None
        self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.sampling_rate = sampling_rate
        self.wavelet_families = ['db1', 'db2', 'db3', 'db4']
        self.is_trained = False

    def normalize_signal(self, data):
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            signal = data[i]
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val - min_val == 0:
                normalized_data[i] = np.zeros_like(signal)
            else:
                normalized_data[i] = (signal - min_val) / (max_val - min_val)
        return normalized_data

    def preprocess_signal(self, data, target_sampling_rate=64):
        # Remove DC component (remove mean)
        data = data - np.mean(data, axis=1, keepdims=True)

        # Apply bandpass filter (0.5 - 20 Hz)
        nyquist = self.sampling_rate / 2
        low, high = 0.5 / nyquist, 20 / nyquist 
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 1, data)

        # Calculate the M
        downsample_factor = int(self.sampling_rate / target_sampling_rate)

        # Apply Low pass filter to avoid aliasing
        nyquist_downsampled = target_sampling_rate / 2
        low_pass_cutoff = nyquist_downsampled / self.sampling_rate

        # Error handling for low_pass_cutoff
        if low_pass_cutoff <= 0 or low_pass_cutoff >= 1:
            low_pass_cutoff = 0.99
        b_low, a_low = signal.butter(4, low_pass_cutoff, btype='low')
        filtered_for_downsampling = np.apply_along_axis(lambda x: signal.filtfilt(b_low, a_low, x), 1, filtered_data)

        # Downsample the Signal
        downsampled_data = filtered_for_downsampling[:, ::downsample_factor]

        # return the normalized signal
        return self.normalize_signal(downsampled_data)

    def extract_features(self, data):
        features = []
        for signal in data:
            signal_features = []
            for wavelet in self.wavelet_families:
                coeffs = pywt.wavedec(signal, wavelet, level=4)
                for coeff in coeffs:
                    signal_features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
            features.append(signal_features)
        return np.array(features)

    def load_data(self, filename):
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
        labels = [label] * num_samples
        self.labels = self.label_encoder.fit_transform(labels)

    def train_model(self, data, labels):
        if len(data) != len(labels):
            raise ValueError("The number of samples and labels must be the same.")

        preprocessed_data = self.preprocess_signal(data)
        features = self.extract_features(preprocessed_data)

        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        self.label_encoder.fit(np.unique(labels))
        encoded_labels = self.label_encoder.transform(labels)

        self.model.fit(scaled_features, encoded_labels)
        self.is_trained = True

        accuracy = self.model.score(scaled_features, encoded_labels)
        return accuracy, preprocessed_data

    def predict(self, test_data):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")

        preprocessed_data = self.preprocess_signal(test_data)
        features = self.extract_features(preprocessed_data)
        scaled_features = self.scaler.transform(features)

        predictions = self.model.predict(scaled_features)
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        return decoded_predictions

    def evaluate_model(self, test_data, true_labels):
        predictions = self.predict(test_data)
        return predictions  


































