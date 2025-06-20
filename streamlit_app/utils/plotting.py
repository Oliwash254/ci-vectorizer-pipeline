import matplotlib.pyplot as plt
import numpy as np

def plot_electrodogram(pulse_times, pulse_amplitudes, reverse=False, small=False):
    """
    Plot an electrodogram given pulse times and amplitudes for each channel.

    Parameters:
    - pulse_times: list of np.arrays, each containing pulse times for a channel
    - pulse_amplitudes: list of np.arrays, each containing amplitudes for a channel
    - reverse: bool, whether to reverse channel order (e.g., for Cochlear systems)
    - small: bool, whether to render a compact version of the plot

    Returns:
    - matplotlib Figure object
    """
    n_channels = len(pulse_times)
    fig, ax = plt.subplots(figsize=(8, 6) if not small else (3, 3))

    for ch in range(n_channels):
        times = pulse_times[ch]
        amps = pulse_amplitudes[ch]

        if len(times) == 0 or len(amps) == 0:
            continue  # skip empty channels

        ch_idx = n_channels - ch - 1 if reverse else ch
        y = np.full_like(times, ch_idx, dtype=np.float32)

        sc = ax.scatter(
            times,
            y,
            c=amps,
            cmap="viridis",
            vmin=0,
            vmax=np.percentile(amps, 99),
            s=1 if small else 5
        )

    if reverse:
        ax.invert_yaxis()

    fig.colorbar(sc, ax=ax, label="Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title("Electrodogram")
    ax.set_ylim(-1, n_channels)
    fig.tight_layout()

    return fig
