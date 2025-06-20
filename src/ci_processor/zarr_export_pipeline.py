import os
import zipfile
import numpy as np
import zarr
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from ci_processor.vectorizer.vectorizer import ConstantPulseVectorizer, ConstantPulseVectorizerClr
from ci_processor.vectorizer.conversion import zip_to_npdict


def extract_metadata(wav_path):
    data, fs = sf.read(wav_path)
    duration = len(data) / fs
    return {
        "fs": fs,
        "wav_duration": duration,
        "wav_filename": os.path.basename(wav_path),
        "raw_wav": data if len(data) < 1_000_000 else None  # avoid huge arrays
    }

def compute_pulse_stats(pulse_times, wav_duration):
    pulses_per_channel = np.array([len(ch) for ch in pulse_times])
    n_pulses = pulses_per_channel.sum()
    pulse_rate = n_pulses / wav_duration
    pulse_rate_per_channel = pulses_per_channel / wav_duration
    return {
        "n_pulses": int(n_pulses),
        "mean_rate": float(pulse_rate),
        "pulses_per_channel": pulses_per_channel.tolist(),
        "pulse_rate_per_channel": pulse_rate_per_channel.tolist()
    }

def process_recording(zip_path, wav_path, system_type, zarr_root, rel_group_path):
    # Extract raw data
    npdict = zip_to_npdict(zip_path)
    fs_scope = npdict['__info__']['fs_scope']

    # Get audio metadata
    meta = extract_metadata(wav_path)
    duration = meta["wav_duration"]

    # Choose processor
    X = np.array([npdict['seg1'][list(npdict['seg1'].keys())[0]][ch] for ch in sorted(npdict['seg1'][list(npdict['seg1'].keys())[0]])])
    if system_type.lower() == "cochlear":
        processor = ConstantPulseVectorizerClr(X, fs_scope)
    else:
        processor = ConstantPulseVectorizer(X, fs_scope)

    pulse_prms = processor.fit_pulse_shape(*processor.average_pulse_shape())
    pulse_times, pulse_amps, _ = processor.vectorize(pulse_prms)

    # Compute additional metadata
    meta.update({
        "system_type": system_type,
        "pulse_prms": pulse_prms.tolist(),
    })
    meta.update(compute_pulse_stats(pulse_times, duration))

    # Save to Zarr
    group = zarr_root.require_group(rel_group_path)
    group.attrs.update(meta)
    group.create_dataset("pulse_times", data=np.array(pulse_times, dtype=object), object_codec=zarr.Blosc())
    group.create_dataset("pulse_amplitudes", data=np.array(pulse_amps, dtype=object), object_codec=zarr.Blosc())
    group.create_dataset("pulse_prms", data=pulse_prms)

def walk_and_process(root_dir, zarr_out_path):
    zarr_root = zarr.open_group(zarr_out_path, mode="w")

    for dirpath, _, filenames in tqdm(list(os.walk(root_dir))):
        wavs = [f for f in filenames if f.lower().endswith(".wav") or f.lower().endswith(".flac")]
        zips = [f for f in filenames if f.lower().endswith(".zip")]
        
        if not wavs or not zips:
            continue

        for zip_file in zips:
            base = zip_file.replace(".zip", "")
            wav_file = next((w for w in wavs if w.startswith(base)), None)
            if wav_file is None:
                print(f"[!] Missing matching .wav for: {zip_file}")
                continue

            full_zip = os.path.join(dirpath, zip_file)
            full_wav = os.path.join(dirpath, wav_file)
            rel_path = Path(dirpath).relative_to(root_dir)
            
            system = "cochlear" if "cochlear" in str(rel_path).lower() else "ab"
            group_path = str(rel_path / base).replace(os.sep, "/")
            try:
                process_recording(full_zip, full_wav, system, zarr_root, group_path)
            except Exception as e:
                print(f"[!] Error processing {zip_file}: {e}")
                continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vectorize CI recordings into Zarr format")
    parser.add_argument("--input", type=str, required=True, help="Path to root directory with recordings")
    parser.add_argument("--output", type=str, required=True, help="Output path for Zarr file")
    args = parser.parse_args()

    walk_and_process(args.input, args.output)
