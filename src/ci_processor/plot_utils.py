import numpy as np
import matplotlib.pyplot as plt
import os


def save_array_preview(array_dict, rec_key, output_dir="output/plots"):
    """
    Generate and save a simple preview plot for the extracted electrodogram arrays.

    Parameters:
        array_dict (dict): The dictionary containing extracted pulse data, e.g.:
            {
                'pulse_times': np.ndarray,
                'pulse_amplitudes': np.ndarray,
                'pulse_prms': np.ndarray
            }
        rec_key (str): Recording key for filename labeling.
        output_dir (str): Directory to save the preview plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    pulse_times = array_dict.get("pulse_times")
    pulse_amplitudes = array_dict.get("pulse_amplitudes")

    if pulse_times is None or pulse_amplitudes is None:
        print(f"[WARNING] Missing pulse_times or pulse_amplitudes for {rec_key}. Skipping preview.")
        return

    plt.figure(figsize=(10, 6))

    for ch in range(len(pulse_times)):
        if len(pulse_times[ch]) == 0:
            continue
        plt.scatter(pulse_times[ch], np.full_like(pulse_times[ch], ch + 1),  # channels start at 1
                    c=pulse_amplitudes[ch], cmap='viridis', s=5, alpha=0.8)

    plt.colorbar(label="Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title(f"Preview Electrodogram - {rec_key}")
    plt.grid(alpha=0.3)

    output_path = os.path.join(output_dir, f"preview_{rec_key}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[✓] Saved preview plot: {output_path}")
