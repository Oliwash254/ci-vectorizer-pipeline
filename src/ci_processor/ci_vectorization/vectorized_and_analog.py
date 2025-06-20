import numpy as np
import npdict
import matplotlib.pyplot as plt
import soundfile as sf
from ci_processor.ci_vectorization.vectorizers import ConstantPulseVectorizerClr, ConstantPulseVectorizerAB

def detect_system_type(zip_path):
    zip_path_lower = zip_path.lower()
    if "cochlear" in zip_path_lower:
        return "cochlear"
    elif "ab" in zip_path_lower:
        return "ab"
    else:
        raise ValueError("Unknown system type in zip path.")

def get_wav_duration(wav_path):
    info = sf.info(wav_path)
    return info.duration  

def plot_analog_and_vectorized(t, X, pulse_times, pulse_amplitudes, recording_name="Recording", system="CI"):
    max_pulse = max([max(L) if len(L) > 0 else 0 for L in pulse_amplitudes])
    n_channels = len(pulse_times)

    plt.figure(figsize=(12, 8))

    # Determine channel order depending on system
    if system.lower() == "cochlear":
        channel_indices = list(reversed(range(n_channels)))  # Reverse channel order
    else:
        channel_indices = list(range(n_channels))  # Normal order

    # Analog signal (light violet), commented out to only show vectorized pulses
    #for plot_idx, ch_idx in enumerate(channel_indices):
        #plt.plot(t, plot_idx+1 + np.array(X[ch_idx])/(max_pulse*2), color='violet', linewidth=0.3, alpha=0.6)

    # Vectorized pulses (strong navy)
    for plot_idx, ch_idx in enumerate(channel_indices):
        plt.vlines(pulse_times[ch_idx], plot_idx+1, plot_idx+1 + np.array(pulse_amplitudes[ch_idx])/(max_pulse*2), 
                   color='navy', linewidth=0.5, alpha=1.0)

    # Gridlines — horizontal (channels)
    for i in range(1, n_channels + 1):
        plt.axhline(i, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)

    # Gridlines — vertical (time)
    xticks = np.arange(0, t[-1], step=0.5)
    for x in xticks:
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)

    if system.lower() == "cochlear":
        plt.yticks(ticks=np.arange(1, n_channels + 1), labels=list(reversed(range(1, n_channels + 1))))
    else:
        plt.yticks(ticks=np.arange(1, n_channels + 1), labels=np.arange(1, n_channels + 1))
    plt.xlabel('Time (s)')
    plt.ylabel('Channel number')

    # Pretty system name
    system_names = {"cochlear": "Cochlear", "ab": "Advanced Bionics"}
    pretty_system = system_names.get(system.lower(), system)
    plt.title(f"{recording_name} ({pretty_system}): Vectorized Electrodogram")

    plt.tight_layout()
    plt.show()

# provide paths to ZIP and WAV files
zip_path = r"C:\TESTING\Recordings_AB\Emotionally expressive speech (EmoHI)\t1_sad\AB_recording_20250522_2057.zip"
wav_path = r"C:\TESTING\Recordings_AB\Emotionally expressive speech (EmoHI)\t1_sad\t1_sad_s1_u06.wav"
rec_key = "t1_sad_s1_u06"

# Load data from ZIP
dat = npdict.zip_to_npdict(zip_path)
sig = dat['rec'][rec_key]
X = np.array(list(sig.values()))
fs = dat['__info__']['fs_scope']
t = np.arange(len(X[0])) / fs

# Trim to match WAV length
wav_duration = get_wav_duration(wav_path)
n_samples = int(wav_duration * fs)
X = X[:, :n_samples]
t = t[:n_samples]

# Detect CI system
system = detect_system_type(zip_path)
if system == "cochlear":
    V = ConstantPulseVectorizerClr(X, fs)
elif system == "ab":
    V = ConstantPulseVectorizerAB(X, fs)

# Vectorize
pulse_times, pulse_amplitudes, pulse_prms = V.vectorize()

# Print arrays
print("Pulse Times:")
print(pulse_times)
print("\nPulse Amplitudes:")
print(pulse_amplitudes)
print("\nPulse Parameters:")
print(pulse_prms)

# Plot
plot_analog_and_vectorized(t, X, pulse_times, pulse_amplitudes, recording_name=rec_key, system=system)