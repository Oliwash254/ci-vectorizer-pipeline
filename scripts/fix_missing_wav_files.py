import os
import shutil
from pathlib import Path
from difflib import get_close_matches

def find_best_wav_match(zip_stem, wav_list):
    """
    Find the best-matching WAV file based on partial name similarity.
    """
    matches = get_close_matches(zip_stem, [w.stem for w in wav_list], n=1, cutoff=0.5)
    if matches:
        return next((w for w in wav_list if w.stem == matches[0]), None)
    return None

def fix_missing_wav_files(root_dir):
    """
    Traverse the dataset. For each .zip file, find a matching .wav file
    (based on stem similarity) and copy it into the same directory if missing.
    """
    print(f"[~] Scanning for missing .wav files in: {root_dir}")

    # Collect all .wav files in the dataset (flattened)
    all_wavs = list(Path(root_dir).rglob("*.wav"))

    if not all_wavs:
        print("[!] No .wav files found in the dataset.")
        return

    n_fixed = 0
    for zip_path in Path(root_dir).rglob("*.zip"):
        zip_stem = zip_path.stem
        folder = zip_path.parent
        expected_wav = folder / f"{zip_stem}.wav"

        if expected_wav.exists():
            continue  # Already present

        best_match = find_best_wav_match(zip_stem, all_wavs)
        if best_match:
            shutil.copy2(best_match, expected_wav)
            print(f"[✓] Copied matching .wav: {best_match} → {expected_wav}")
            n_fixed += 1
        else:
            print(f"[!] No matching .wav found for: {zip_path.name}")

    print(f"\n[✓] Completed. Fixed {n_fixed} missing .wav files.")

if __name__ == "__main__":
    DATA_ROOT = "data/Bscproject_library"  # Change if needed
    fix_missing_wav_files(DATA_ROOT)
