# ... [unchanged imports at top]
import zipfile
import json
import os
import numpy as np
import pickle
import numbers
from pathlib import Path

# ... [unchanged _save_array, _load_array]

def _load_item(zf, item_zip_path_stem):
    print(f"DEBUG: _load_item called for item_zip_path_stem: '{item_zip_path_stem}'")

    try:
        full_path = f"{item_zip_path_stem}.npy.meta"
        with zf.open(full_path, 'r') as f_meta:
            print(f"DEBUG: Found {full_path}")
            return _load_array(f_meta)
    except KeyError:
        pass

    for ext in [".info.json", ".json"]:
        json_full_path = f"{item_zip_path_stem}{ext}"
        try:
            with zf.open(json_full_path, 'r') as f_json:
                print(f"DEBUG: Found {json_full_path}")
                json_data = json.load(f_json)

                if isinstance(json_data, dict) and 'dtype' in json_data and 'shape' in json_data:
                    dat_full_path = f"{item_zip_path_stem}.dat"
                    try:
                        with zf.open(dat_full_path, 'r') as f_dat:
                            print(f"DEBUG: Found and loading {dat_full_path} with metadata from {json_full_path}")
                            dtype = np.dtype(json_data['dtype'])
                            shape = tuple(json_data['shape'])
                            data_bytes = f_dat.read()
                            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                    except KeyError:
                        print(f"DEBUG: {dat_full_path} not found for metadata {json_full_path}, returning JSON only.")
                        return json_data
                else:
                    print(f"DEBUG: {json_full_path} is a standalone JSON, returning it.")
                    return json_data
        except KeyError:
            continue

    # Try .pkl or .pickle
    for ext in [".pkl", ".pickle"]:
        try:
            pkl_full_path = f"{item_zip_path_stem}{ext}"
            with zf.open(pkl_full_path) as f_pkl:
                print(f"DEBUG: Found {pkl_full_path}")
                return pickle.load(f_pkl)
        except KeyError:
            pass

    print(f"DEBUG: _load_item failed: No recognizable data file found for stem '{item_zip_path_stem}' in zip.")
    raise FileNotFoundError(f"No recognizable data file found for stem '{item_zip_path_stem}' in zip.")


def _load_npdict_from_zip_recursive(zf, current_zip_dir_path=""):
    print(f"DEBUG: _load_npdict_from_zip_recursive called for current_zip_dir_path: '{current_zip_dir_path}'")
    result = {}
    processed_children = {}

    for name_in_zip in zf.namelist():
        if not name_in_zip.startswith(current_zip_dir_path):
            continue

        relative_part = name_in_zip[len(current_zip_dir_path):]
        if not relative_part:
            continue

        parts = relative_part.split('/', 1)
        immediate_child_exact_zip_name = parts[0]
        display_key_base = immediate_child_exact_zip_name.strip('"')

        if len(parts) > 1:
            exact_dir_path_for_recursion = current_zip_dir_path + immediate_child_exact_zip_name + '/'
            processed_children[display_key_base] = (exact_dir_path_for_recursion, True)
        else:
            item_stem_for_loader = None
            if immediate_child_exact_zip_name.endswith('.npy.meta'):
                item_stem_for_loader = immediate_child_exact_zip_name.rsplit('.', 2)[0]
            elif immediate_child_exact_zip_name.endswith('.info.json'):
                item_stem_for_loader = immediate_child_exact_zip_name.replace('.info.json', '')
            elif immediate_child_exact_zip_name.endswith('.json'):
                item_stem_for_loader = immediate_child_exact_zip_name.rsplit('.', 1)[0]
            elif immediate_child_exact_zip_name.endswith('.pkl') or immediate_child_exact_zip_name.endswith('.pickle'):
                item_stem_for_loader = immediate_child_exact_zip_name.rsplit('.', 1)[0]
            elif immediate_child_exact_zip_name.endswith('.dat'):
                continue
            else:
                print(f"DEBUG: Skipping unrecognized file type: '{name_in_zip}'")
                continue

            item_display_key = item_stem_for_loader.strip('"')
            full_path_stem_for_load_item = current_zip_dir_path + item_stem_for_loader

            if item_display_key not in processed_children:
                processed_children[item_display_key] = (full_path_stem_for_load_item, False)

    print(f"DEBUG: Processed children info for '{current_zip_dir_path}': {processed_children}")

    for display_key, (exact_path_for_next_step, is_directory) in processed_children.items():
        if is_directory:
            result[display_key] = _load_npdict_from_zip_recursive(zf, exact_path_for_next_step)
        else:
            try:
                result[display_key] = _load_item(zf, exact_path_for_next_step)
            except FileNotFoundError as e:
                print(f"Warning: Could not load item for key '{display_key}' at path stem '{exact_path_for_next_step}': {e}")

    return result


def zip_to_npdict(filename, prefer_unzipped=False):
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    if prefer_unzipped:
        folder_root = filename.parent / filename.stem
        actual_rec_dir = folder_root / 'rec'
        actual_info_dir = folder_root / '__info__'

        if actual_rec_dir.exists() and actual_rec_dir.is_dir():
            print(f"Loading from unzipped folder: {folder_root}")
            return folder_to_npdict(actual_rec_dir, info_dir=actual_info_dir if actual_info_dir.exists() else None)

        print(f"Unzipped folder not found or incomplete, extracting {filename}...")
        try:
            with zipfile.ZipFile(filename, 'r') as zf:
                zf.extractall(folder_root)
            print(f"Extracted {filename} to {folder_root}")
            return folder_to_npdict(actual_rec_dir, info_dir=actual_info_dir if actual_info_dir.exists() else None)
        except Exception as e:
            print(f"Error extracting ZIP file: {e}. Attempting to load directly from ZIP.")
            with zipfile.ZipFile(filename, 'r') as zf:
                return _load_npdict_from_zip_recursive(zf)
    else:
        with zipfile.ZipFile(filename, 'r') as zf:
            return _load_npdict_from_zip_recursive(zf)


def folder_to_npdict(directory, info_dir=None):
    data = {"rec": {}, "__info__": {}}
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Provided path is not a directory: {directory}")

    # Load metadata from __info__
    if info_dir and info_dir.is_dir():
        for f in info_dir.iterdir():
            if f.suffix == ".json":
                try:
                    with open(f, 'r') as json_f:
                        data["__info__"][f.stem] = json.load(json_f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {f.name}")
            elif f.suffix in [".pkl", ".pickle"]:
                try:
                    with open(f, 'rb') as pkl_f:
                        data["__info__"][f.stem] = pickle.load(pkl_f)
                except (pickle.UnpicklingError, EOFError):
                    print(f"Warning: Could not unpickle from {f.name}")
            elif f.suffix == ".npy":
                try:
                    data["__info__"][f.stem] = np.load(f)
                except Exception as e:
                    print(f"Warning: Could not load numpy array from {f.name}: {e}")

    # Load recording segments
    for item in directory.iterdir():
        if item.is_dir():
            try:
                segment_key = item.name
                data["rec"][segment_key] = {}
                for channel_file in item.iterdir():
                    if channel_file.suffix == ".npy":
                        channel_key = channel_file.stem
                        data["rec"][segment_key][channel_key] = np.load(channel_file)
                    elif channel_file.suffix == ".dat":
                        base_name = channel_file.stem
                        json_file = channel_file.parent / f"{base_name}.json"
                        if not json_file.exists():
                            json_file = channel_file.parent / f"{base_name}.info.json"
                        if json_file.exists():
                            with open(json_file, 'r') as f_json:
                                json_data = json.load(f_json)
                            dtype = np.dtype(json_data['dtype'])
                            shape = tuple(json_data['shape'])
                            with open(channel_file, 'rb') as f_dat:
                                data_bytes = f_dat.read()
                                data["rec"][segment_key][base_name] = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                    elif channel_file.suffix in [".pkl", ".pickle"]:
                        try:
                            with open(channel_file, 'rb') as pkl_f:
                                channel_key = channel_file.stem
                                data["rec"][segment_key][channel_key] = pickle.load(pkl_f)
                        except (pickle.UnpicklingError, EOFError) as e:
                            print(f"Warning: Could not unpickle from {channel_file.name} in segment {segment_key}: {e}")
            except Exception as e:
                print(f"Warning: Could not process directory {item.name} as a segment: {e}")

    return data
