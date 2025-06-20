# cochlear_vectorizer.py
import numpy as np
# Assuming ConstantPulseVectorizer and oversample_recording_matrix are in the same package (ci_vectorization)
from .utils import ConstantPulseVectorizer, oversample_recording_matrix


class ConstantPulseVectorizerClr(ConstantPulseVectorizer):
    """
    Extracts pulse timings and amplitudes from Cochlear analog recordings.
    Designed for ~500 kHz sampled data.
    Inherits from ConstantPulseVectorizer to reuse common logic.
    """

    anodic_first = False # Cochlear pulses are typically cathodic first (negative leading phase)
    # Cochlear systems often operate with pulse widths in microseconds.
    # While there isn't a strict "base period" like AB's 0.898 us, the resolution
    # of their stimulators is often in ~0.1 - 1 us range. We'll keep a dummy for now.
    # If a precise value is known, it should be set here.
    base_period = 0.5e-6 # Dummy value for quantization if needed, refine based on actual CLR specs.

    def __init__(self, X, fs, channel_labels=None, oversample_ratio=None):
        """
        Parameters
        ----------
        X : numpy.ndarray
            An n*m array, with n the number of channels, and m the number of samples per channel.
        fs : float
            The scope's sampling frequency.
        channel_labels : list, optional
            If not provided, uses a list of 1 to `X.shape[0]`.
        oversample_ratio : int, optional
            If omitted or None, calculates a ratio to ensure sufficiently high sampling rate
            for pulse fitting (e.g., at least 2x `1/base_period`).
        """
        # Ensure X is a 2D array, even if a single channel is passed as 1D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.oversample_ratio = 1

        if oversample_ratio is None:
            # Calculate oversample_ratio to ensure the new sampling rate is at least 2 times the inverse of base_period
            # This is to help with fitting resolution, even if Cochlear doesn't have a strict "base_period"
            target_fs = 2 / self.base_period # Aim for at least 2 samples per dummy base period
            if fs < target_fs:
                oversample_ratio = int(np.ceil(target_fs / fs))
            else:
                oversample_ratio = 1 # No oversampling needed if current fs is already high enough
        
        if oversample_ratio > 1:
            # print(f"Cochlear Vectorizer: Data will be oversampled {oversample_ratio} times (from {fs} Hz to {fs*oversample_ratio} Hz).")
            X_oversampled, fs_oversampled = oversample_recording_matrix(X, fs, n=oversample_ratio)
            self.oversample_ratio = oversample_ratio
            super().__init__(X_oversampled, fs_oversampled, anodic_first=self.anodic_first, channel_labels=channel_labels)
        else:
            super().__init__(X, fs, anodic_first=self.anodic_first, channel_labels=channel_labels)

    def quantize_pulse_shape(self, pulse_prms):
        """
        Quantizes the phase durations and inter-phase gap if needed.
        For Cochlear, this might be less critical than AB, but included for consistency.
        """
        # pulse_prms format: [t0, p1, ipg, p2, tau, a]
        # We need to quantize p1, ipg, p2 (indices 1, 2, 3)
        quantized_prms = np.array(pulse_prms).copy()
        
        # Quantize p1, ipg, p2 (if base_period is meaningful for CLR)
        for i in [1, 2, 3]:
            quantized_prms[i] = np.round(pulse_prms[i] / self.base_period) * self.base_period
        
        # Ensure positive values after quantization if any become negative due to rounding small values
        for i in [1, 2, 3, 4]: # Also ensure tau is positive
            quantized_prms[i] = max(0, quantized_prms[i])

        return quantized_prms

    def fit_pulse_shape(self, *args, **kwargs):
        """
        Overrides the base class method to add relevant constraints for Cochlear.
        Cochlear pulses are typically symmetric biphasic with no inter-phase gap.
        """
        if 'extra_constraints' not in kwargs or kwargs['extra_constraints'] is None:
            kwargs['extra_constraints'] = []

        # Add quantization constraint (if base_period is meaningful for CLR)
        kwargs['extra_constraints'].append(self.quantize_pulse_shape)

        # Cochlear pulses are typically symmetric, enforce this by default.
        if 'sym_pulse' not in kwargs:
            kwargs['sym_pulse'] = True
        
        # Cochlear pulses typically have no IPG, enforce this by default.
        if 'no_ipg' not in kwargs:
             kwargs['no_ipg'] = True

        # Call the base class fit_pulse_shape with the added constraint
        result = super().fit_pulse_shape(*args, **kwargs)

        if isinstance(result, tuple):
            return result
        else:
            return result

    def find_pulses_corr(self, pulse_prms, X=None, fs=None, thr=None, force_sequential=True):
        """
        Locates pulses using the pulse shape correlation method. Default `force_sequential`
        to True as CIS is common for Cochlear devices.

        Parameters
        ----------
        pulse_prms : list
            Pulse parameters, as returned by :meth:`fit_pulse_shape`.
        X : numpy.ndarray, optional
            The recording signal matrix. If omitted, `self.X` is used.
        fs : float, optional
            The sampling frequency. If omitted, `self.fs` is used.
        thr : float, optional
            The threshold value. If None, then the value is determined from the histogram.
        force_sequential : bool, default=True
            Whether to ensure that pulses follow each other (like in CIS).
            Default is True for Cochlear.

        Returns
        -------
        peaks : list
            For each channel, a list of peak indices.
        peak_info : list
            Details of the identified peaks (including their amplitude) (a `dict` per channel).
        p : numpy.ndarray
            Pulse shape used in the correlation.
        t_pulse : numpy.ndarray
            The time vector associated with the pulse shape.
        thr : float
            The computed threshold that was used (if thr was None).
        """
        return super().find_pulses_corr(pulse_prms, X, fs, thr, force_sequential)


# This function acts as the entry point for vectorize in vectorizers.py
# It will return an instance of ConstantPulseVectorizerClr, which then has the .vectorize() method.
def vectorize_cochlear(X, fs) -> ConstantPulseVectorizerClr:
    """
    Returns an instantiated ConstantPulseVectorizerClr object ready for vectorization.

    Parameters
    ----------
    X : numpy.ndarray
        An n*m array, with n the number of channels, and m the number of samples per channel.
    fs : float
        The scope's sampling frequency.

    Returns
    -------
    ConstantPulseVectorizerClr
        An instantiated vectorizer object.
    """
    return ConstantPulseVectorizerClr(X=X, fs=fs)