import os
import numpy as np
import soundfile as sf
import zipfile
import zarr
from tqdm import tqdm
from datetime import datetime

from ci_processor.vectorizer.vectorizer import ConstantPulseVectorizerClr, ConstantPulseVectorizerAB
from ci_processor.vectorizer.conversion import sparse_to_long
from ci_processor.elpi_electric_recording.npdict import zip_to_npdict


def detect_system_type(zip_path):
    name = os.path.basename(zip_path).lower()
    if "cochlear" in name:
        return "Cochlear"
    elif "ab" in name or "advanced_bionics" in name:
        return "AB"
    else:
        return None


def extract_npdict(zip_path):
    return zip_to_npdict(zip_path)


def extract_pulse_info(npdict, system_type):
    fs = npdict["__info__"]["fs_scope"]
    X = npdict["rec"]
    channels = list(sorted(X.keys()))
    X_arr = np.vstack([X[ch] for ch in channels])

    if system_type == "Cochlear":
        v = ConstantPulseVectorizerClr(X_arr, fs)
    elif system_type == "AB":
        v = ConstantPulseVectorizerAB(X_arr, fs)
    else:
        raise ValueError("Unknown system type")

    pulse_prms = v.fit_pulse_shape()
    pulse_times, pulse_amplitudes = v.vectorize(*pulse_prms)

    return pulse_times, pulse_amplitudes, pulse_prms, fs


def compute_metadata(pulse_times, fs, wav_duration, wav_filename, system_type):
    n_pulses = sum(len(ch) for ch in pulse_times)
    mean_rate = n_pulses / wav_duration if wav_duration > 0 else 0

    pulses_per_ch = np.array([len(ch) for ch in pulse_times])
    pps_per_ch = pulses_per_ch / wav_duration if wav_duration > 0 else np.zeros_like(pulses_per_ch)

    return {
        "fs": fs,
        "wav_duration": wav_duration,
        "wav_filename": wav_filename,
        "system_type": system_type,
        "n_pulses": int(n_pulses),
        "mean_pulse_rate": float(mean_rate),
        "pulses_per_channel": pulses_per_ch,
        "pulse_rate_per_channel": pps_per_ch,
    }


def export_to_zarr(zarr_root, rel_path, pulse_times, pulse_amplitudes, pulse_prms, metadata):
    grp = zarr_root.require_group(rel_path)

    grp.array("pulse_times", np.array(pulse_times, dtype=object), object_codec=zarr.Blosc())
    grp.array("pulse_amplitudes", np.array(pulse_amplitudes, dtype=object), object_codec=zarr.Blosc())
    grp.array("pulse_prms", pulse_prms)

    for k, v in metadata.items():
        if isinstance(v, (np.ndarray, list)):
            grp.attrs[k] = np.array(v).tolist()
        else:
            grp.attrs[k] = v


def process_single_folder(folder_path, zarr_root):
    zip_path = next((os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".zip")), None)
    wav_path = next((os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")), None)

    if zip_path is None or wav_path is None:
        return

    rel_path = os.path.relpath(folder_path)

    try:
        wav_data, wav_fs = sf.read(wav_path)
        wav_duration = len(wav_data) / wav_fs
    except:
        wav_duration = 0

    system_type = detect_system_type(zip_path)
    npdict = extract_npdict(zip_path)
    pulse_times, pulse_amplitudes, pulse_prms, fs = extract_pulse_info(npdict, system_type)

    metadata = compute_metadata(
        pulse_times, fs, wav_duration, os.path.basename(wav_path), system_type
    )

    export_to_zarr(zarr_root, rel_path, pulse_times, pulse_amplitudes, pulse_prms, metadata)


def run_pipeline(root_folder, zarr_output_path):
    store = zarr.DirectoryStore(zarr_output_path)
    root = zarr.group(store=store, overwrite=True)

    for dirpath, dirnames, filenames in tqdm(os.walk(root_folder)):
        if any(f.endswith(".zip") for f in filenames) and any(f.endswith(".wav") for f in filenames):
            process_single_folder(dirpath, root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CI recording vectorization pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to root folder of recordings")
    parser.add_argument("--output", type=str, required=True, help="Output path for zarr folder")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
