# EOG Project - Readme

## Preprocessing

## DC Removal
Subtract the mean of each signal to center it around zero. This reduces the sharpening of the signal.

## Bandpass Filtering
Apply a Butterworth filter to retain frequencies between 0.5 Hz and 20 Hz. This removes noise and irrelevant components.

## Normalization
Scale values from 0 to 1.

## Resampling
Downsample the signal. Remove high frequencies before downsampling using a low-pass filter.

## Feature Extraction

## Purpose
Decompose signals into components at different frequency bands.

## Wavelet Family
Use the Daubechies wavelet (db4).

## Levels
Apply four levels of decomposition to capture frequencies ranging from 0.5 Hz to 20 Hz.

## Final KNN Model Result

- Training Accuracy: 100%
- Test Accuracy: 90%

