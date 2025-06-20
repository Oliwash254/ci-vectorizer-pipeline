# ci_processor/ci_vectorization/vectorizers.py
import numpy as np
import scipy as sp
import scipy.signal as sps
import scipy.fft as spf
import scipy.linalg as spl
from scipy.optimize import minimize
import pandas as pd
from . import conversion
import math

from .ab_vectorizer import vectorize_ab, ConstantPulseVectorizerAB
from .cochlear_vectorizer import vectorize_cochlear, ConstantPulseVectorizerClr
from . import cochlear_vectorizer
from . import ab_vectorizer

SYSTEM_AB = "AB"
SYSTEM_COCHLEAR = "cochlear"

SUPPORTED_SYSTEMS = {
    SYSTEM_AB.lower(): SYSTEM_AB,
    SYSTEM_COCHLEAR.lower(): SYSTEM_COCHLEAR
}

# MODIFIED: get_vectorizer to return instances of the vectorizer classes
def get_vectorizer(system_type: str):
    system_type = system_type.strip().lower()

    if system_type in {"ab", "advancedbionics"}:
        return ConstantPulseVectorizerAB() # Return instance
    elif system_type in {"cochlear", "clr"}:
        return ConstantPulseVectorizerClr() # Return instance

    raise ValueError(f"Unknown system type: {system_type}")

# The `vectorize` function below is no longer used by main_pipeline.py
# as it now calls the vectorize method on the returned vectorizer instance.
def vectorize(npdict_data, system): # Renamed 'npdict' parameter to 'npdict_data' to avoid confusion with the module
    info = npdict_data.get("__info__", {})
    fs_scope = info.get("fs_scope") or info.get("fs", 0)
    if fs_scope == 0:
        raise ValueError("Sampling frequency not found in __info__.")

    rec = npdict_data.get("rec", {})
    if not rec:
        raise ValueError("No recording data found in 'rec' field.")

    # Assuming 'rec' contains segments as direct children, and each segment contains channels
    # This structure needs to match what npdict.py provides
    all_pulse_times = {}
    all_pulse_amplitudes = {}

    for segment_key, segment_data in rec.items():
        if not isinstance(segment_data, dict) or not all(isinstance(v, np.ndarray) for v in segment_data.values()):
            print(f"Warning: Skipping segment '{segment_key}' due to unexpected data structure.")
            continue
        
        # Stack channel data into a 2D array (n_channels, n_samples)
        # Assuming channel keys are numeric strings that can be sorted
        sorted_channel_keys = sorted(segment_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
        X_list = [segment_data[ch_key] for ch_key in sorted_channel_keys]
        
        # Pad to max length if channels have different lengths
        max_len = max(len(arr) for arr in X_list)
        X_padded = [np.pad(arr, (0, max_len - len(arr))) for arr in X_list]
        X = np.array(X_padded)

        # Get the appropriate vectorized function
        if system.lower() == SYSTEM_AB.lower():
            segment_pulse_times, segment_pulse_amplitudes = vectorize_ab(X, fs_scope)
        elif system.lower() == SYSTEM_COCHLEAR.lower():
            segment_pulse_times, segment_pulse_amplitudes = vectorize_cochlear(X, fs_scope)
        else:
            raise ValueError(f"Unsupported system type for vectorization: {system}")

        all_pulse_times[segment_key] = segment_pulse_times
        all_pulse_amplitudes[segment_key] = segment_pulse_amplitudes

    return all_pulse_times, all_pulse_amplitudes

# The classes for vectorization (ConstantPulseVectorizerClr, ConstantPulseVectorizerAB)
# are assumed to be defined elsewhere or in the imported cochlear_vectorizer and ab_vectorizer.
# The user's provided file actually contains them at the end. I will keep them here.

# Assume ConstantPulseVectorizerClr and ConstantPulseVectorizerAB classes are defined within
# cochlear_vectorizer.py and ab_vectorizer.py respectively, and are imported.
# If they are in this file, their definitions should follow.
# Based on the user's provided vectorizers.py, these classes are defined directly within it.
# Keeping them here to match the user's provided file structure.

class ConstantPulseVectorizerClr:
    """
    Constant Pulse Vectorizer for Cochlear CI systems.
    Converts continuous neural signals into discrete pulse events
    based on a thresholding and pulse shaping model specific to Cochlear.
    """
    def __init__(self):
        # Initialize any parameters specific to Cochlear vectorization
        self.conversion_factor = 1.0 # Example parameter
        # Assuming Cochlear pulses are biphasic by default:
        self.pulse_shape_params = {'p1': 0.05e-3, 'ipg': 0.02e-3, 'p2': 0.05e-3, 'tau1': 0.1e-3, 'tau2': 0.5e-3, 'ntau': 5}

    def vectorize(self, X: np.ndarray, fs: float):
        """
        Vectorize the input signal X into pulse times and amplitudes.
        X: 2D numpy array of shape (n_channels, n_samples)
        fs: Sampling frequency
        Returns: tuple of (list of pulse times, list of pulse amplitudes)
        """
        pulse_times = []
        pulse_amplitudes = []
        pulse_prms = [] # To store pulse parameters if needed

        # Example: Simple thresholding and constant pulse amplitude for illustration
        # This is a placeholder; actual vectorization would involve more complex algorithms
        threshold = 0.1 # Example threshold
        # Assuming pulse_amplitudes are derived from X values that exceed a threshold
        # And pulse_times are the sample indices converted to time

        for i in range(X.shape[0]): # Iterate over channels
            channel_signal = X[i, :]
            
            # Simple peak detection (for demonstration)
            peaks, _ = sps.find_peaks(channel_signal, height=threshold)
            
            # Convert sample indices to time
            times = peaks / fs
            
            # Assign constant amplitude for simplicity, or derive from signal strength
            amplitudes = channel_signal[peaks] # Use actual signal amplitude at peak

            pulse_times.append(times)
            pulse_amplitudes.append(amplitudes)
            # For this simple example, pulse_prms would be empty or a list of placeholder dicts
            pulse_prms.append([{} for _ in times]) 

        return pulse_times, pulse_amplitudes, pulse_prms

    def pulse(self, t, t0, a, p1, ipg, p2, tau1, tau2):
        # Placeholder for pulse shape generation
        # This function should generate a single pulse shape
        v = np.zeros_like(t, dtype=float)
        
        s = (t >= t0) & (t <= t0 + p1)
        if tau1 != 0:
            v[s] = (1 - np.exp(-(t[s] - t0) / tau1)) - 1
        else:
            v[s] = -1  # Assign a default value if tau1 is zero
        ve = v[s][-1] if np.any(s) else 0

        s = (t >= t0 + p1) & (t <= t0 + p1 + ipg)
        if np.any(s):
            if tau1 != 0:
                v[s] = ve * np.exp(-(t[s] - t0 - p1) / tau1)
            else:
                v[s] = ve
            ve = v[s][-1]

        s = (t >= t0 + p1 + ipg) & (t <= t0 + p1 + ipg + p2)
        if np.any(s):
            v[s] = 1 - (1 - ve) * np.exp(-(t[s] - t0 - p1 - ipg) / tau1)
            ve = v[s][-1]

        s = t >= t0 + p1 + ipg + p2
        if np.any(s):
            if tau2 != 0:
                v[s] = ve * np.exp(-(t[s] - t0 - p1 - ipg - p2) / tau2)
            else:
                v[s] = ve # Assign the value directly if tau2 is zero

        return v * a
    
    def fit_pulse_shape(self, *args, **kwargs):
        if 'ntau' not in kwargs:
            kwargs['ntau'] = self.pulse_shape_params['ntau']
        # Placeholder for fitting pulse shape parameters
        # This would involve optimizing the parameters to match a target pulse shape
        return self.pulse_shape_params # Return default for now

class ConstantPulseVectorizerAB:
    """
    Constant Pulse Vectorizer for Advanced Bionics CI systems.
    Similar to Cochlear but with potentially different pulse shaping parameters
    and processing logic specific to AB.
    """
    def __init__(self):
        # Initialize any parameters specific to AB vectorization
        self.conversion_factor = 0.5 # Example parameter
        # Assuming AB pulses are triphasic by default:
        self.pulse_shape_params = {'p1': 0.04e-3, 'ipg': 0.01e-3, 'p2': 0.04e-3, 'tau1': 0.1e-3, 'tau2': 0.5e-3, 'ntau': 5}

    def vectorize(self, X: np.ndarray, fs: float):
        """
        Vectorize the input signal X into pulse times and amplitudes.
        X: 2D numpy array of shape (n_channels, n_samples)
        fs: Sampling frequency
        Returns: tuple of (list of pulse times, list of pulse amplitudes)
        """
        pulse_times = []
        pulse_amplitudes = []
        pulse_prms = [] # To store pulse parameters if needed

        # Example: Simple thresholding and constant pulse amplitude for illustration
        # This is a placeholder; actual vectorization would involve more complex algorithms
        threshold = 0.05 # Example threshold

        for i in range(X.shape[0]): # Iterate over channels
            channel_signal = X[i, :]
            
            # Simple peak detection (for demonstration)
            peaks, _ = sps.find_peaks(channel_signal, height=threshold)
            
            # Convert sample indices to time
            times = peaks / fs
            
            # Assign constant amplitude for simplicity, or derive from signal strength
            amplitudes = channel_signal[peaks] # Use actual signal amplitude at peak

            pulse_times.append(times)
            pulse_amplitudes.append(amplitudes)
            pulse_prms.append([{} for _ in times]) 

        return pulse_times, pulse_amplitudes, pulse_prms
    
    def pulse(self, t, t0, a, p1, ipg, p2, tau1, tau2):
        # Placeholder for pulse shape generation for AB
        v = np.zeros_like(t, dtype=float)
        
        s = (t >= t0) & (t <= t0 + p1)
        if tau1 != 0:
            v[s] = (1 - np.exp(-(t[s] - t0) / tau1)) - 1
        else:
            v[s] = -1  # Assign a default value if tau1 is zero
        ve = v[s][-1] if np.any(s) else 0

        s = (t >= t0 + p1) & (t <= t0 + p1 + ipg)
        if np.any(s):
            if tau1 != 0:
                v[s] = ve * np.exp(-(t[s] - t0 - p1) / tau1)
            else:
                v[s] = ve
            ve = v[s][-1]

        s = (t >= t0 + p1 + ipg) & (t <= t0 + p1 + ipg + p2)
        if np.any(s):
            v[s] = 1 - (1 - ve) * np.exp(-(t[s] - t0 - p1 - ipg) / tau1)
            ve = v[s][-1]

        s = t >= t0 + p1 + ipg + p2
        if np.any(s):
            if tau2 != 0:
                v[s] = ve * np.exp(-(t[s] - t0 - p1 - ipg - p2) / tau2)
            else:
                v[s] = ve # Assign the value directly if tau2 is zero

        return v * a
    
    def fit_pulse_shape(self, *args, **kwargs):
        if 'ntau' not in kwargs:
            kwargs['ntau'] = self.pulse_shape_params['ntau']
        # Placeholder for fitting pulse shape parameters
        return self.pulse_shape_params # Return default for now