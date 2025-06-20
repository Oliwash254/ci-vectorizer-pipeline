from pathlib import Path
import os

root_dir = Path("data/Bscproject_library")

renamed = []

# Traverse all subfolders
for zip_file in root_dir.rglob("*.zip"):
    zip_stem = zip_file.stem  # e.g., AB_recording_20250522_2057
    folder = zip_file.parent

    # List all WAVs in the same folder
    wav_files = list(folder.glob("*.wav"))

    for i, wav in enumerate(wav_files):
        new_name = f"{zip_stem}_{i+1}.wav"
        new_path = wav.with_name(new_name)
        os.rename(wav, new_path)
        renamed.append((wav.name, new_name))  # ✅ Fixed: closed parenthesis

# Log what we renamed
print(f"\n✅ Renamed {len(renamed)} .wav files:\n")
for old, new in renamed:
    print(f"{old}  →  {new}")
