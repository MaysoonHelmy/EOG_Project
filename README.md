Preprocessing
- DC Removal
Subtract the mean of each signal to center it around zero , To reduce the sharpening of the signal
- Bandpass Filtering
Apply a Butterworth filter to retain frequencies between 0.5 Hz and 20 Hz, removing noise and irrelevant components.
- Normalization
Scaling from 0 to 1
- Resampling by downsampling , remove high frequencies before down sampling by low pass filter

Feature Extraction
- Purpose: Decompose signals into components at different frequency bands
- Wavelet Family : db4
- Levels: 4 levels to get the range from 0.5 to 20

Final KNN Model Result 
- Training Accuracy : 100 % ,
- Test Accuracy : 90 %
