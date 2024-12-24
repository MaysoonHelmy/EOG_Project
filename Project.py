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
        """Normalize the signal to [0, 1] range."""
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
        """Preprocess the raw signals by applying mean removal, bandpass filtering, normalization, and either upsampling or downsampling."""
        # DC removal
        data = data - np.mean(data, axis=1, keepdims=True)

        # Bandpass filter (0.5-20 Hz) - Applying bandpass filter before upsampling or downsampling
        nyquist = self.sampling_rate / 2
        low, high = 0.5 / nyquist, 20 / nyquist  # Normalized cutoff frequency (between 0 and 1)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 1, data)

        if self.sampling_rate < target_sampling_rate:
            # Upsampling Logic
            upsample_factor = int(target_sampling_rate / self.sampling_rate)

            # Apply a low-pass filter to prevent aliasing before upsampling
            nyquist_upsampled = target_sampling_rate / 2
            low_pass_cutoff = nyquist_upsampled / self.sampling_rate

            # Ensure that the cutoff is within the valid range (0 < cutoff < 1)
            if low_pass_cutoff <= 0 or low_pass_cutoff >= 1:
                low_pass_cutoff = 0.99  # Set to a safe value close to Nyquist to prevent aliasing

            b_low, a_low = signal.butter(4, low_pass_cutoff, btype='low')
            filtered_for_upsampling = np.apply_along_axis(lambda x: signal.filtfilt(b_low, a_low, x), 1, filtered_data)

            # Upsample by interpolating between original samples
            upsampled_data = np.apply_along_axis(lambda x: np.interp(np.arange(0, len(x), 1/upsample_factor), np.arange(0, len(x)), x), 1, filtered_for_upsampling)

            return self.normalize_signal(upsampled_data)

        elif self.sampling_rate > target_sampling_rate:
            # Downsampling Logic
            downsample_factor = int(self.sampling_rate / target_sampling_rate)

            # Apply a low-pass filter to prevent aliasing before downsampling
            nyquist_downsampled = target_sampling_rate / 2
            low_pass_cutoff = nyquist_downsampled / self.sampling_rate

            # Ensure that the cutoff is within the valid range (0 < cutoff < 1)
            if low_pass_cutoff <= 0 or low_pass_cutoff >= 1:
                low_pass_cutoff = 0.99  # Set to a safe value close to Nyquist to prevent aliasing

            b_low, a_low = signal.butter(4, low_pass_cutoff, btype='low')
            filtered_for_downsampling = np.apply_along_axis(lambda x: signal.filtfilt(b_low, a_low, x), 1, filtered_data)

            # Downsample by picking every nth sample
            downsampled_data = filtered_for_downsampling[:, ::downsample_factor]

            return self.normalize_signal(downsampled_data)

        else:
            # If the target_sampling_rate is equal to the original sampling rate, just normalize the data
            return self.normalize_signal(filtered_data)

    def extract_features(self, data):
        """Extract wavelet-based features from the signals."""
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

        # Fit the scaler and transform features
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)

        # Encode labels
        self.label_encoder.fit(np.unique(labels))
        encoded_labels = self.label_encoder.transform(labels)

        # Train the KNN model
        self.model.fit(scaled_features, encoded_labels)
        self.is_trained = True

        # Return model accuracy
        accuracy = self.model.score(scaled_features, encoded_labels)
        return accuracy, preprocessed_data

    def predict(self, test_data):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")

        # Preprocess and extract features
        preprocessed_data = self.preprocess_signal(test_data)
        features = self.extract_features(preprocessed_data)
        scaled_features = self.scaler.transform(features)

        predictions = self.model.predict(scaled_features)
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        return decoded_predictions

    def evaluate_model(self, test_data, true_labels):
        """Evaluate the model on test data."""
        predictions = self.predict(test_data)
        return predictions  # Only return predictions, excluding accuracy and other metrics
