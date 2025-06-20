import os
import zipfile
import shutil
import argparse
import numpy as np
import soundfile as sf
import zarr
import traceback
import datetime
from tqdm import tqdm
from pathlib import Path
from difflib import get_close_matches
import json

from numcodecs import JSON

# --- NEW: Import actual npdict functions ---
# Assuming npdict.py is in the same 'ci_processor' directory or accessible via PYTHONPATH
# If it's in ci_processor, the import would be:
from ci_processor.npdict import folder_to_npdict, zip_to_npdict
# If it's in a different higher-level directory, you might need to adjust the import path.
# For now, let's assume it's directly importable from ci_processor.

# --- START MOCK OBJECTS FOR TESTING (RETAINING VECTORIZER MOCKS FOR NOW) ---
class MockVectorizer:
    def __call__(self, npdict, rec_key):
        arrays = {}
        if "waveform" in npdict:
            arrays["waveform"] = npdict["waveform"]
        
        if "rec" in npdict and rec_key in npdict["rec"]:
            rec_data_entry = npdict["rec"][rec_key]
            # Your real npdict structure might be different here.
            # The mock assumed {"data": np.random.rand(...)}.
            # Your real npdict.py creates {"rec": {"rec_key": {"ch_key": np.ndarray}}}.
            # We need to adapt the mock or assume the vectorizer expects a certain structure.
            # For now, let's assume rec_data_entry is directly the numpy array for simplicity,
            # but this will need careful review when integrating real vectorizers.
            
            # --- IMPORTANT: Adapt mock to match REAL npdict structure ---
            # Your real npdict.py returns:
            # {
            #     "rec": {
            #         "rec_key": {
            #             "ch_key_1": np.ndarray,
            #             "ch_key_2": np.ndarray,
            #             ...
            #         }
            #     },
            #     "__info__": {...}
            # }
            # So, rec_data_entry will be a dictionary like {"ch_key": np.ndarray}.
            # Let's mock a simple output from multiple channels if available, or a default.

            if isinstance(rec_data_entry, dict) and rec_data_entry: # Check if it's a non-empty dictionary
                # Take the first channel's data as a basis for shape, or just a mock
                first_channel_data = next(iter(rec_data_entry.values()))
                if isinstance(first_channel_data, np.ndarray) and first_channel_data.ndim >= 1:
                    # Simulate electrodogram output based on the number of samples
                    # For a single channel input, maybe 10 electrodes x num_samples
                    arrays["electrodogram"] = np.random.rand(10, first_channel_data.shape[-1])
                else:
                    arrays["electrodogram"] = np.random.rand(10, 100) # Default if data structure is unexpected
            else:
                arrays["electrodogram"] = np.random.rand(10, 100) # Default if no channels or empty dict
            # --- END Adaptation ---
        
        arrays["processed_data"] = np.random.rand(100)
        return arrays

class MockVectorizerModule:
    @staticmethod
    def get_vectorizer(system):
        return MockVectorizer()

vz = MockVectorizerModule()

# These are now placeholders for the real plotting functions
def save_waveform_plot(data, key, output_folder):
    pass
def save_electrodogram_plot(data, key, output_folder):
    pass
# --- END MOCK OBJECTS ---


def detect_system_type(path):
    lower_path = str(path).lower()
    if "cochlear" in lower_path:
        return "Cochlear"
    if "ab" in lower_path:
        return "AB"
    for p in path.parents:
        lower_p = str(p).lower()
        if "cochlear" in lower_p:
            return "Cochlear"
        if "ab" in lower_p:
            return "AB"
    return None

def safe_json_attrs(attrs: dict):
    cleaned = {}
    for key, val in attrs.items():
        if isinstance(val, np.ndarray):
            cleaned[key] = val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            cleaned[key] = val.item()
        else:
            cleaned[key] = val
    return cleaned

def find_all_recordings(input_dir: Path):
    candidates = []

    for path in input_dir.rglob("*.zip"):
        if path.stem.lower().startswith("ab_recording_") or path.stem.lower().startswith("cochlear_recording_"):
            candidates.append(path)

    for path in input_dir.rglob("*"):
        if not path.is_dir():
            continue

        if path.name.lower() in {"_rec_", "__rec__", "_info_", "__info__", "rec", "info"}:
            continue
        
        if (path.name.lower().startswith("ab_recording_") or path.name.lower().startswith("cochlear_recording_")):
            if (path / "_rec_").exists() or \
               (path / "__rec__").exists() or \
               (path / "rec").exists() or \
               (path / "___info___").exists() or \
               (path / "__info__").exists() or \
               list(path.glob("*.dat")): # Check for 'rec' without underscores. Keep comments on their own lines.
                if path not in candidates:
                    candidates.append(path)
                    
    candidates = sorted(list(set(candidates)))
    return candidates

def find_all_wavs(input_dir: Path):
    return list(input_dir.rglob("*.wav"))

def process_input(input_dir: Path, output_path: Path, include_wavs=True, prefer_unzipped=False, allow_missing_wav=False):
    all_wav_files = find_all_wavs(input_dir)
    recordings = find_all_recordings(input_dir)
    zroot = zarr.open(str(output_path), mode='w')

    for item in tqdm(recordings, desc="Processing recordings"):
        try:
            system = detect_system_type(item)
            if not system:
                print(f"[!] {item.name:30} - Unrecognized system, skipping.")
                continue

            rec_base_name = item.stem if item.is_file() else item.name
            
            main_wav_for_this_item_path = None
            
            potential_wavs = [w for w in all_wav_files if w.stem == rec_base_name]
            for wav in potential_wavs:
                if wav.parent == item.parent or wav.parent == item:
                    main_wav_for_this_item_path = wav
                    break
            
            if not main_wav_for_this_item_path:
                potential_segmented_wavs = [w for w in all_wav_files if w.stem.startswith(f"{rec_base_name}_")]
                for wav in potential_segmented_wavs:
                    if wav.parent == item.parent or wav.parent == item:
                        main_wav_for_this_item_path = wav
                        break
            
            if not main_wav_for_this_item_path and item.is_dir():
                potential_wavs_in_parent = [w for w in all_wav_files if (w.parent == item.parent and w.stem == rec_base_name)]
                if potential_wavs_in_parent:
                    main_wav_for_this_item_path = potential_wavs_in_parent[0]
                elif not main_wav_for_this_item_path:
                    potential_segmented_wavs_in_parent = [w for w in all_wav_files if (w.parent == item.parent and w.stem.startswith(f"{rec_base_name}_"))]
                    if potential_segmented_wavs_in_parent:
                        main_wav_for_this_item_path = potential_segmented_wavs_in_parent[0]


            full_wav_data = None
            full_wav_sr = None
            if main_wav_for_this_item_path and include_wavs:
                try:
                    full_wav_data, full_wav_sr = sf.read(main_wav_for_this_item_path)
                    print(f"  Loaded main WAV: {main_wav_for_this_item_path.name} for {rec_base_name}")
                except Exception as e:
                    print(f"[ERROR] Could not load WAV {main_wav_for_this_item_path.name}: {e}")
                    if not allow_missing_wav:
                        continue
            elif include_wavs and not allow_missing_wav:
                 print(f"[WARNING] No main WAV found for recording base: {rec_base_name} at path {item.relative_to(input_dir)}. Skipping processing.")
                 continue


            npdict_data = {} # Renamed from 'npdict' to avoid conflict with imported function
            if item.suffix == ".zip":
                npdict_data = zip_to_npdict(item, prefer_unzipped=prefer_unzipped)
            elif item.is_dir():
                actual_rec_dir = None
                possible_rec_subdirs = ["_rec_", "__rec__", "rec"]
                for subdir_name in possible_rec_subdirs:
                    if (item / subdir_name).exists():
                        actual_rec_dir = item / subdir_name
                        break
                
                if actual_rec_dir is None and list(item.glob("*.dat")):
                    actual_rec_dir = item
                elif actual_rec_dir is None:
                    print(f"[!] {item.name}: No 'rec' subdirectory found (e.g., _rec_, __rec__, rec) or direct .dat files. Skipping.")
                    continue

                actual_info_dir = None
                possible_info_subdirs = ["___info___", "__info__", "info"]
                for subdir_name in possible_info_subdirs:
                    if (item / subdir_name).exists():
                        actual_info_dir = item / subdir_name
                        break

                npdict_data = folder_to_npdict(actual_rec_dir, info_dir=actual_info_dir if actual_info_dir and actual_info_dir.exists() else None)
            else:
                print(f"[!] Skipping unknown item type: {item}, not a zip or recognized directory.")
                continue

            # --- Handle the "__info__" key from the real npdict ---
            # Your real npdict.py returns a dictionary with "__info__" key.
            # The original code expected "metadata". We need to map this.
            # It's better to store __info__ as a direct attribute or combine if needed.
            # For simplicity, let's just make sure the 'rec' key is present.
            
            if not npdict_data.get("rec"):
                print(f"[!] No 'rec' data found in {item.name}. Skipping.")
                continue

            # The real npdict.py might not have "metadata" directly.
            # Let's adjust where metadata is pulled from.
            # The '__info__' key from npdict.py is likely the metadata.
            metadata_from_npdict = npdict_data.get("__info__", {})
            if "fs" in npdict_data:
                metadata_from_npdict["fs"] = npdict_data["fs"]
            
            rec_keys = list(npdict_data["rec"].keys())
            vectorizer = vz.get_vectorizer(system) # Still using mock vectorizer for now

            main_rec_zgroup = zroot.require_group(rec_base_name)
            main_rec_zgroup.attrs["system"] = system
            main_rec_zgroup.attrs.update(safe_json_attrs(metadata_from_npdict)) # Use the new metadata
            main_rec_zgroup.attrs["original_source_path"] = str(item.relative_to(input_dir))
            
            if full_wav_data is not None and include_wavs:
                if "waveform" not in main_rec_zgroup:
                    main_rec_zgroup.create_dataset(name="waveform", data=full_wav_data, dtype=full_wav_data.dtype, overwrite=True)
                    main_rec_zgroup.attrs["audio_sr"] = full_wav_sr
                    save_waveform_plot(full_wav_data, rec_base_name, output_folder="output/plots")
            
            for rec_key in rec_keys:
                try:
                    # Adapt sub_npdict to match what mock vectorizer expects
                    # The mock vectorizer expects `npdict["rec"][rec_key]` to be something it can process
                    # (like a dict with a "data" key, or directly the numpy array).
                    # Your real npdict has {"rec_key": {"ch_key": np.ndarray}}.
                    # For now, let's pass the whole segment's data as is to the mock,
                    # and the mock itself has been slightly adapted to handle it.
                    
                    sub_npdict = {
                        "rec": {rec_key: npdict_data["rec"][rec_key]}, # Pass the nested dict for this segment
                        "metadata": metadata_from_npdict.copy(), # Use the new metadata
                        "fs": npdict_data.get("fs", 500000) # Ensure fs is propagated
                    }

                    if full_wav_data is not None:
                        sub_npdict["waveform"] = full_wav_data
                        sub_npdict["metadata"]["audio_sr"] = full_wav_sr
                    
                    arrays = vectorizer(sub_npdict, rec_key=rec_key)

                    segment_zgroup = main_rec_zgroup.require_group(rec_key)
                    segment_zgroup.attrs["system"] = system 
                    segment_zgroup.attrs["segment_key"] = rec_key
                    segment_zgroup.attrs.update(safe_json_attrs(sub_npdict.get("metadata", {})))

                    for key, arr in arrays.items():
                        if key == "waveform":
                            continue
                        segment_zgroup.create_dataset(name=key, data=arr, dtype=arr.dtype, overwrite=True)

                        if key.startswith("electro"):
                            save_electrodogram_plot(arr, f"{rec_base_name}_{rec_key}", output_folder="output/plots")

                except Exception as sub_e:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [X] {item.name} -> {rec_key} - Error: {sub_e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [X] {item.name:30} - Error: {e}")
            traceback.print_exc()

def main(input_path: str, output_zarr: str, include_wavs: bool, prefer_unzipped: bool, allow_missing_wav: bool):
    input_dir = Path(input_path)
    output_path = Path(output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path("output/plots").mkdir(parents=True, exist_ok=True)

    process_input(
        input_dir=input_dir,
        output_path=output_path,
        include_wavs=include_wavs,
        prefer_unzipped=prefer_unzipped,
        allow_missing_wav=allow_missing_wav
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio recordings into Zarr format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing recordings.")
    parser.add_argument("--output", type=str, required=True, help="Path for the output Zarr store.")
    parser.add_argument("--include_wavs", action="store_true", help="Whether to include waveform data.")
    parser.add_argument("--prefer_unzipped", action="store_true", help="Prefer unzipped folders over zip files if both exist.")
    parser.add_argument("--allow_missing_wav", action="store_true", help="Allow processing even if matching WAV files are missing (will skip items without WAVs if --include_wavs is true).")
    args = parser.parse_args()

    main(
        input_path=args.input_dir,
        output_zarr=args.output,
        include_wavs=args.include_wavs,
        prefer_unzipped=args.prefer_unzipped,
        allow_missing_wav=args.allow_missing_wav
    )