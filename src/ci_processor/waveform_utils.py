import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_waveform_plot(wav, sr, save_path):
    """
    Save a simple waveform plot for the given audio signal.
    """
    plt.figure(figsize=(10, 3))
    time_axis = np.arange(len(wav)) / sr
    plt.plot(time_axis, wav, color='steelblue')
    plt.title("Audio Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_electrodogram_plot(pulses, save_path, cochlear_reverse=False):
    """
    Save an electrodogram heatmap plot for the given pulse matrix.
    """
    plt.figure(figsize=(8, 6))
    
    if cochlear_reverse:
        pulses = np.flipud(pulses)

    vmax = np.percentile(np.abs(pulses), 99)
    plt.imshow(pulses, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    plt.colorbar(label="Amplitude")
    plt.title("Electrodogram")
    plt.ylabel("Channel")
    plt.xlabel("Pulse index")
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
