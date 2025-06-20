from pathlib import Path
from difflib import get_close_matches
import json

data_dir = Path("data/Bscproject_library")

zips = sorted([f for f in data_dir.rglob("*.zip")])
wavs = sorted([f for f in data_dir.rglob("*.wav")])

zip_stems = [z.stem for z in zips]
wav_stems = [w.stem for w in wavs]

mapping = {}

print(f"Found {len(zips)} ZIPs and {len(wavs)} WAVs.\n")

for zip_file in zips:
    zip_stem = zip_file.stem
    match = get_close_matches(zip_stem, wav_stems, n=1, cutoff=0.5)
    if match:
        wav_match = next((w for w in wavs if w.stem == match[0]), None)
        mapping[zip_stem] = wav_match.name
        print(f"[MATCH] {zip_file.name}  ⇨  {wav_match.name}")
    else:
        print(f"[NO MATCH] {zip_file.name}")

# Optionally save to file
with open("wav_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)

print("\n✅ Saved matching results to wav_mapping.json")
