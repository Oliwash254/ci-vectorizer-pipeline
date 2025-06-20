# ab_vectorizer.py
import numpy as np
# Assuming ConstantPulseVectorizer is in the same package (ci_vectorization)
from .utils import ConstantPulseVectorizer, oversample_recording_matrix


class ConstantPulseVectorizerAB(ConstantPulseVectorizer):
    """
    A fixed pulse shape vectorizer for AB recordings.
    Inherits from ConstantPulseVectorizer to reuse common logic.
    """

    anodic_first = True # AB pulses are typically anodic first (positive leading phase)
    base_period = 0.898e-6 # The smallest time interval used in AB devices (0.898 μs).

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
            If omitted or None, a value will be calculated such that the new sampling rate is higher than
            the the one corresponding to the base period (0.898 μs).
        """
        # Ensure X is a 2D array, even if a single channel is passed as 1D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.oversample_ratio = 1

        if oversample_ratio is None:
            # Calculate oversample_ratio to ensure the new sampling rate is at least 2 times the inverse of base_period
            # (or higher, depending on precision needed for pulse fitting)
            target_fs = 2 / self.base_period # Aim for at least 2 samples per base period
            if fs < target_fs:
                oversample_ratio = int(np.ceil(target_fs / fs))
            else:
                oversample_ratio = 1 # No oversampling needed if current fs is already high enough
        
        if oversample_ratio > 1:
            # print(f"AB Vectorizer: Data will be oversampled {oversample_ratio} times (from {fs} Hz to {fs*oversample_ratio} Hz).")
            X_oversampled, fs_oversampled = oversample_recording_matrix(X, fs, n=oversample_ratio)
            self.oversample_ratio = oversample_ratio
            super().__init__(X_oversampled, fs_oversampled, anodic_first=self.anodic_first, channel_labels=channel_labels)
        else:
            super().__init__(X, fs, anodic_first=self.anodic_first, channel_labels=channel_labels)
        

    def quantize_pulse_shape(self, pulse_prms):
        """
        Quantizes the phase durations and inter-phase gap to multiples of the base period.
        """
        # pulse_prms format: [t0, p1, ipg, p2, tau, a]
        # We need to quantize p1, ipg, p2 (indices 1, 2, 3)
        quantized_prms = np.array(pulse_prms).copy()
        
        # Quantize p1, ipg, p2
        for i in [1, 2, 3]:
            quantized_prms[i] = np.round(pulse_prms[i] / self.base_period) * self.base_period
        
        # Ensure positive values after quantization if any become negative due to rounding small values
        for i in [1, 2, 3, 4]: # Also ensure tau is positive
            quantized_prms[i] = max(0, quantized_prms[i])

        return quantized_prms

    def fit_pulse_shape(self, *args, **kwargs):
        """
        Overrides the base class method to add quantization constraint.
        """
        if 'extra_constraints' not in kwargs or kwargs['extra_constraints'] is None:
            kwargs['extra_constraints'] = []

        # Add the quantization constraint
        kwargs['extra_constraints'].append(self.quantize_pulse_shape)

        # AB pulses are typically symmetric, enforce this by default if not specified.
        if 'sym_pulse' not in kwargs:
            kwargs['sym_pulse'] = True
        
        # AB pulses typically have no IPG for standard pulses, enforce this by default if not specified.
        # This is a common assumption for basic AB pulses. Can be set to False if a specific AB
        # pulse type *does* have an IPG.
        if 'no_ipg' not in kwargs:
             kwargs['no_ipg'] = True

        # Call the base class fit_pulse_shape with the added constraint
        # The base class fit_pulse_shape returns (params, prms_per_pulse) if multiple pulses are fit
        # or just params if a single average pulse is fit.
        result = super().fit_pulse_shape(*args, **kwargs)

        # If the result is a tuple (mean_params, all_params), return it as is.
        # Otherwise, it's just the mean_params.
        if isinstance(result, tuple):
            return result
        else:
            return result # This will be the single array of pulse parameters

    def find_pulses_corr(self, pulse_prms, X=None, fs=None, thr=None, force_sequential=False):
        """
        Locates pulses using the pulse shape correlation method. Like :meth:`ConstantPulseVectorizer.find_pulses_corr`
        but with `force_sequential` as False by default for AB, as AB often uses stimulation strategies
        where pulses might not be strictly sequential across all channels (e.g., interleaved pulsatile stimulation).

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
        force_sequential : bool, default=False
            Whether to ensure that pulses follow each other (like in CIS).
            Default is False for AB.

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
# It will return an instance of ConstantPulseVectorizerAB, which then has the .vectorize() method.
def vectorize_ab(X, fs) -> ConstantPulseVectorizerAB:
    """
    Returns an instantiated ConstantPulseVectorizerAB object ready for vectorization.

    Parameters
    ----------
    X : numpy.ndarray
        An n*m array, with n the number of channels, and m the number of samples per channel.
    fs : float
        The scope's sampling frequency.

    Returns
    -------
    ConstantPulseVectorizerAB
        An instantiated vectorizer object.
    """
    return ConstantPulseVectorizerAB(X=X, fs=fs)