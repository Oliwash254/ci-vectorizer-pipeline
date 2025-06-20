# vectorized_recording.py
"""
Script for visualizing cochlear implant electrodograms from ZIP and WAV files.

This script loads a ZIP file containing multi-channel analog recordings
(e.g., from a cochlear implant processor) and a corresponding WAV file.
It performs the following steps:

1. Loads the analog recording and metadata from the ZIP using npdict.
2. Identifies the correct recording key based on the WAV filename (or first available).
3. Trims the recording to match the duration of the WAV file.
4. Detects the implant system type (Cochlear or Advanced Bionics).
5. Uses the appropriate vectorizer to extract:
    - Pulse times (s)
    - Pulse amplitudes (µA or normalized)
    - Optional fitted pulse parameters
6. Plots the resulting electrodogram as a vertical line plot (one line per pulse, per channel).
7. Prints the pulse time and amplitude arrays for inspection.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Correct relative imports based on the project structure
from ci_processor.npdict import zip_to_npdict # Correct import for npdict function
from ci_processor.ci_vectorization.vectorizers import get_vectorizer # Import the get_vectorizer factory function


def detect_system_type_from_path(path):
    """Detects the CI system type based on path name."""
    lower_path = str(path).lower()
    if "cochlear" in lower_path:
        return "Cochlear" # Return "Cochlear" as string for consistency with get_vectorizer
    elif "ab" in lower_path:
        return "AB" # Return "AB" as string for consistency with get_vectorizer
    else:
        # Fallback or raise error if unable to determine
        print(f"Warning: Could not determine system type from path: {path}. Defaulting to 'Cochlear'.")
        return "Cochlear" # Defaulting for this example, adjust as needed

def get_wav_duration(wav_path):
    """Gets the duration of a WAV file in seconds."""
    info = sf.info(wav_path)
    return info.duration

def plot_electrodogram(pulse_times, pulse_amplitudes, system_type, rec_name):
    """
    Plots the electrodogram as vertical lines representing pulses.

    Parameters
    ----------
    pulse_times : list of numpy.ndarray
        List of 1D arrays, where each array contains pulse times for a channel (in seconds).
    pulse_amplitudes : list of numpy.ndarray
        List of 1D arrays, where each array contains pulse amplitudes for a channel.
    system_type : str
        Type of CI system ('Cochlear' or 'AB') for plot title.
    rec_name : str
        Name of the recording for plot title.
    """
    n_channels = len(pulse_times)

    # Calculate max amplitude across all channels for normalization, handle empty channels
    all_amplitudes = np.concatenate([ch_amps for ch_amps in pulse_amplitudes if len(ch_amps) > 0])
    max_amp = np.max(np.abs(all_amplitudes)) if all_amplitudes.size > 0 else 1.0 # Avoid division by zero

    fig, ax = plt.subplots(figsize=(12, 6)) # Create figure and axes
    
    for ch_idx in range(n_channels):
        if len(pulse_times[ch_idx]) > 0: # Only plot if pulses exist for the channel
            # For plotting, offset each channel's pulses vertically
            # np.array(pulse_amplitudes[ch_idx]) / (2 * max_amp) scales amplitudes
            # such that they don't overlap too much, assuming max_amp is positive.
            # (ch_idx + 1) is the baseline for each channel.
            ax.vlines(pulse_times[ch_idx], ch_idx + 1, ch_idx + 1 + pulse_amplitudes[ch_idx] / (2 * max_amp),
                       color='blue', linewidth=0.5, alpha=0.7) # Added alpha for density

    ax.set_yticks(ticks=np.arange(1, n_channels + 1)) # Y-ticks for channels (1-based)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"Electrodogram: {rec_name} ({system_type})")
    ax.set_ylim(0.5, n_channels + 0.5) # Set limits to nicely contain all channels
    ax.invert_yaxis() # Often, lower channel numbers are at the top

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set file paths - REPLACE WITH YOUR ACTUAL PATHS FOR TESTING
    zip_path = r"C:\Users\Anita\Downloads\Bachelor's project\TESTING\Recordings_Cochlear\Emotionally expressive speech (EmoHI)\t1_sad\Cochlear_recording_20250513_1429.zip"
    wav_path = r"C:\Users\Anita\Downloads\Bachelor's project\TESTING\Recordings_Cochlear\Emotionally expressive speech (EmoHI)\t1_sad\t1_sad_s1_u06.wav"

    if not os.path.exists(zip_path):
        print(f"Error: ZIP file not found at {zip_path}")
        exit()
    if not os.path.exists(wav_path):
        print(f"Error: WAV file not found at {wav_path}")
        exit()

    print(f"Processing ZIP: {os.path.basename(zip_path)}")
    print(f"Associated WAV: {os.path.basename(wav_path)}")

    # Load npdict from the ZIP file
    dat = zip_to_npdict(zip_path)

    # Determine rec_key for the signal data
    # Assuming for this example script that the 'rec' dict might contain a single key,
    # or that the WAV filename matches a numeric key.
    # In a real scenario, you'd need a more robust way to link WAV to specific recording segment.
    
    rec_keys = list(dat.get('rec', {}).keys())
    if not rec_keys:
        raise ValueError("No recording data found in the ZIP file.")
    
    # Try to match WAV stem to rec_key, otherwise pick the first one
    wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
    rec_key = None
    if wav_stem in rec_keys: # Direct match
        rec_key = wav_stem
    elif len(rec_keys) == 1: # Only one recording available
        rec_key = rec_keys[0]
    else: # Try to find a numeric match if WAV stem is not a direct key
          # This part is heuristic and might need adjustment based on real data conventions
        try:
            # If WAV is 'my_recording_segment_X.wav', maybe 'X' is the key
            potential_numeric_key = wav_stem.split('_')[-1]
            if potential_numeric_key in rec_keys and potential_numeric_key.isdigit():
                rec_key = potential_numeric_key
        except IndexError:
            pass # Not a segmented WAV

        if rec_key is None:
            print(f"Warning: Could not find a specific recording key '{wav_stem}' or numeric match in {rec_keys}. Using the first found key: {rec_keys[0]}")
            rec_key = rec_keys[0]


    # Get the raw signal (electro_array) and sampling frequency
    # Ensure signal is a 2D array (n_channels, n_samples)
    raw_signal_dict = dat['rec'].get(rec_key)
    if raw_signal_dict is None:
        raise KeyError(f"Recording key '{rec_key}' not found in 'dat['rec']'. Available keys: {rec_keys}")
    
    X = np.stack([raw_signal_dict[k] for k in sorted(raw_signal_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)])
    fs = dat.get('fs', 500000) # Use .get with default, consistent with main_pipeline

    t_full_signal = np.arange(X.shape[1]) / fs

    print(f"Loaded signal with {X.shape[0]} channels, {X.shape[1]} samples at {fs} Hz.")

    # Trim the recording to match WAV duration
    duration = get_wav_duration(wav_path)
    n_samples_wav = int(duration * fs)
    
    if n_samples_wav > X.shape[1]:
        print(f"Warning: WAV duration ({duration:.2f}s) is longer than signal ({t_full_signal[-1]:.2f}s). Using full signal duration.")
        n_samples_to_use = X.shape[1]
    else:
        n_samples_to_use = n_samples_wav

    X_trimmed = X[:, :n_samples_to_use]
    t_trimmed = t_full_signal[:n_samples_to_use]

    print(f"Trimmed signal to {X_trimmed.shape[1]} samples ({duration:.2f}s).")

    # Detect system type and get the appropriate vectorizer
    system_type = detect_system_type_from_path(zip_path) # Use system detection from helper function
    vectorizer_class = get_vectorizer(system_type)
    
    # Instantiate and vectorize
    vectorizer_instance = vectorizer_class(X=X_trimmed, fs=fs)
    pulse_times, pulse_amplitudes, pulse_prms = vectorizer_instance.vectorize()

    # Show result
    plot_electrodogram(pulse_times, pulse_amplitudes, system_type, rec_key)
    print("\n--- Vectorization Results ---")
    print("Pulse times (first 5 per channel):")
    for i, pt in enumerate(pulse_times):
        print(f"  Ch {i+1}: {pt[:5]}...")
    print("\nPulse amplitudes (first 5 per channel):")
    for i, pa in enumerate(pulse_amplitudes):
        print(f"  Ch {i+1}: {pa[:5]}...")
    print("\nPulse parameters:", pulse_prms)