import numpy as np

def load_flat_pulse_arrays(recording):
    pulse_times = recording["pulse_times"][:]
    pulse_times_channel = recording["pulse_times_channel"][:]
    pulse_amplitudes = recording["pulse_amplitudes"][:]
    pulse_amplitudes_channel = recording["pulse_amplitudes_channel"][:]

    n_channels = np.max(pulse_times_channel) + 1
    pulse_times_per_channel = [pulse_times[pulse_times_channel == ch] for ch in range(n_channels)]
    pulse_amplitudes_per_channel = [pulse_amplitudes[pulse_amplitudes_channel == ch] for ch in range(n_channels)]

    return pulse_times_per_channel, pulse_amplitudes_per_channel
