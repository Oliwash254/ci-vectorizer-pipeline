import numpy as np
from scipy.signal import find_peaks

class ConstantPulseVectorizer:
    """
    Base class for simple pulse detection.
    Can be subclassed for Cochlear or AB depending on polarity, thresholds, etc.
    """

    def __init__(self, X, fs):
        """
        X: ndarray (channels x samples)
        fs: sampling frequency
        """
        self.X = X
        self.fs = fs
        self.n_channels, self.n_samples = X.shape

    def vectorize(self):
        """
        Main pulse extraction method.
        Returns:
            pulse_times: list of arrays (in seconds)
            pulse_amplitudes: list of arrays
            pulse_params: list of dicts (optional additional parameters)
        """
        pulse_times = []
        pulse_amplitudes = []
        pulse_params = []

        for ch in range(self.n_channels):
            signal = self.X[ch]
            times, amps, params = self._process_channel(signal)
            pulse_times.append(times)
            pulse_amplitudes.append(amps)
            pulse_params.append(params)

        return pulse_times, pulse_amplitudes, pulse_params

    def _process_channel(self, signal):
        """
        Default naive peak detection (can be overridden by subclasses).
        """
        # Very simple peak detection as placeholder
        peaks, _ = find_peaks(signal, height=np.std(signal)*0.5, distance=self.fs * 0.001)
        times = peaks / self.fs
        amplitudes = signal[peaks]
        params = {}
        return times, amplitudes, params
