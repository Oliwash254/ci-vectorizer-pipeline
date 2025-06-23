import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import traceback
import zipfile
import numcodecs # Import numcodecs for object_codec
import json # Import json for serialization
import shutil
import glob # Import glob for pattern matching

# Import functions from your ci_processor library
try:
    from ci_processor.ci_vectorization.npdict import zip_to_npdict, folder_to_npdict
    from ci_processor.ci_vectorization.vectorizers import get_vectorizer, SYSTEM_AB, SYSTEM_COCHLEAR
    from ci_processor.electrodogram import plot_electrodogram

except ImportError as e:
    print(f"[ERROR] Failed to import ci_processor modules. Please ensure your 'ci_processor' "
          f"package is correctly installed and accessible in your Python environment. "
          f"Error: {e}")
    exit(1)


def detect_system_type(path_or_id):
    """
    Detect CI system type from a path string or recording ID.
    Expects 'cochlear' or 'ab' to be present in the lowercased path/ID.
    """
    path_or_id_lower = str(path_or_id).lower()
    if "cochlear" in path_or_id_lower:
        return SYSTEM_COCHLEAR
    elif "ab" in path_or_id_lower:
        return SYSTEM_AB
    else:
        print("[WARN] Could not detect system type from path/ID. Defaulting to 'AB'.")
        return SYSTEM_AB

def get_zarr_recording_group_path(input_item_path, root_dataset_folder):
    """
    Constructs the full Zarr group path for a recording based on its original
    filesystem hierarchy relative to the root_dataset_folder.
    This path will represent the group for the ZIP file/folder itself.

    Example:
    input_item_path = "C:/TESTING/Recordings_AB/Emotionally expressive speech/t1_sad/AB_recording_20250522_2057.zip"
    root_dataset_folder = "C:/TESTING"
    -> Returns "Recordings_AB/Emotionally expressive speech/t1_sad/AB_recording_20250522_2057"
    """
    # Get the directory containing the zip/folder
    input_parent_dir = os.path.dirname(input_item_path)
    
    # Get the basename of the zip/folder without extension
    recording_name_from_item = os.path.splitext(os.path.basename(input_item_path))[0]
    
    # Get the relative path from the root_dataset_folder to the input_parent_dir
    relative_path_to_parent = os.path.relpath(input_parent_dir, root_dataset_folder)
    
    # Combine relative path and recording name, converting backslashes to forward slashes for Zarr
    full_recording_group_path = os.path.join(relative_path_to_parent, recording_name_from_item).replace(os.sep, '/')
    
    return full_recording_group_path


def process_single_recording(input_item_path, root_dataset_folder, root_zarr_store_obj, parent_output_dir_for_root_zarr):
    """
    Processes a single CI recording (ZIP or unzipped folder) and saves its vectorized data
    to the appropriate location within the main Zarr library. Also attempts to find and
    store paths to associated WAV files.

    Args:
        input_item_path (str): Absolute path to a single ZIP file or unzipped recording folder.
        root_dataset_folder (str): The absolute path to the top-level folder of the raw dataset.
        root_zarr_store_obj (zarr.Group): The opened root Zarr store (e.g., BSCProject_LIBRARY.zarr).
        parent_output_dir_for_root_zarr (str): The directory containing the root Zarr store.
    """
    print(f"\n--- Processing recording: {input_item_path} ---")

    # Determine the full hierarchical Zarr group path for THIS recording (e.g., .../AB_recording_XXXX)
    full_recording_group_zarr_path = get_zarr_recording_group_path(input_item_path, root_dataset_folder)
    print(f"Constructed Zarr recording group path: '{full_recording_group_zarr_path}'")

    # Create/Access the specific recording group (for the ZIP file) within the root Zarr store
    try:
        recording_group = root_zarr_store_obj.require_group(full_recording_group_zarr_path)
        print(f"Recording group '{full_recording_group_zarr_path}' created/accessed.")
    except Exception as e:
        print(f"[ERROR] Failed to create/access recording group '{full_recording_group_zarr_path}': {e}")
        print(traceback.format_exc())
        return

    dat = None
    try:
        if os.path.isdir(input_item_path):
            rec_dir = os.path.join(input_item_path, 'rec')
            info_dir = os.path.join(input_item_path, '__info__')
            if not os.path.isdir(rec_dir):
                raise ValueError(f"Missing 'rec' folder in directory: {rec_dir}")
            dat = folder_to_npdict(rec_dir, info_dir=info_dir if os.path.isdir(info_dir) else None)
        elif zipfile.is_zipfile(input_item_path):
            dat = zip_to_npdict(input_item_path, prefer_unzipped=True)
        else:
            raise ValueError("Input item path must be a directory or ZIP file.")
    except Exception as e:
        print(f"[ERROR] Failed to load CI data from '{input_item_path}': {traceback.format_exc()}")
        return

    if not dat or 'rec' not in dat or not dat['rec']:
        print(f"[ERROR] No 'rec' section found in data for '{input_item_path}'. Skipping.")
        return

    try:
        system_type = detect_system_type(input_item_path)
        print(f"Detected system type for '{input_item_path}': {system_type.upper()}")
        vectorizer = get_vectorizer(system_type)
    except Exception as e:
        print(f"[ERROR] Failed to initialize vectorizer for '{input_item_path}': {traceback.format_exc()}")
        return

    info = dat.get('__info__', {})
    fs = info.get('fs_scope') or info.get('fs')

    if fs is None or fs == 0:
        print(f"[ERROR] Sampling rate ('fs_scope' or 'fs') not found or zero for '{input_item_path}'. Skipping.")
        return

    # --- Store __info__ and fs_value directly in the recording_group ---
    if info:
        try:
            info_json_str = json.dumps(info, default=str)
            info_array = np.array([info_json_str], dtype=str)
            zarr.create(
                shape=info_array.shape,
                dtype=info_array.dtype,
                chunks=(1,),
                store=recording_group.store,
                path=f'{recording_group.path}/__info__',
                overwrite=True,
                data=info_array
            )
            print(f"Saved __info__ metadata to Zarr group: '{recording_group.path}/__info__'")
        except Exception as e:
            print(f"[WARN] Could not save __info__ to Zarr for '{recording_group.path}': {e}")
            print(traceback.format_exc())

    if fs is not None:
        try:
            fs_array = np.array([fs], dtype=float)
            zarr.create(
                shape=fs_array.shape,
                dtype=fs_array.dtype,
                chunks=(1,),
                store=recording_group.store,
                path=f'{recording_group.path}/fs_value',
                overwrite=True,
                data=fs_array
            )
            print(f"Saved fs value to Zarr group: '{recording_group.path}/fs_value'")
        except Exception as e:
            print(f"[WARN] Could not save fs to Zarr for '{recording_group.path}': {e}")
            print(traceback.format_exc())

    try:
        from tqdm import tqdm
        segments_iterator = tqdm(dat['rec'].items(), desc=f"Segments in {os.path.basename(input_item_path)}")
    except ImportError:
        segments_iterator = dat['rec'].items()
        print("Install 'tqdm' for a progress bar (pip install tqdm).")

    processed_segments_count = 0
    # Determine the directory where the input ZIP file resides to look for WAVs
    input_item_dir = os.path.dirname(input_item_path)

    # Segments are now direct children of the recording_group, no 'segments' intermediate group
    for segment_name, segment_data in segments_iterator:
        try:
            if not isinstance(segment_data, dict) or not all(isinstance(v, np.ndarray) for v in segment_data.values()):
                raise ValueError("Segment data must be a dict of NumPy arrays.")

            keys_sorted = sorted(segment_data.keys(), key=lambda k: int(k) if k.isdigit() else k)
            X_list = [segment_data[k] for k in keys_sorted]
            max_len = max(len(x) for x in X_list)
            X = np.stack([np.pad(x, (0, max_len - len(x))) for x in X_list])

            pulse_times_channels, pulse_amps_channels, pulse_prms_channels = vectorizer.vectorize(X, fs)

            # --- Prepare data for Zarr saving ---
            max_pulse_events = 0
            if pulse_times_channels and any(len(arr) > 0 for arr in pulse_times_channels):
                max_pulse_events = max(len(arr) for arr in pulse_times_channels)

            pulse_times_to_save = np.array([
                np.pad(arr, (0, max_pulse_events - len(arr)), 'constant', constant_values=np.nan)
                for arr in pulse_times_channels
            ]) if pulse_times_channels else np.empty((0, max(1, max_pulse_events)), dtype=np.float64)

            pulse_amps_to_save = np.array([
                np.pad(arr, (0, max_pulse_events - len(arr)), 'constant', constant_values=np.nan)
                for arr in pulse_amps_channels
            ]) if pulse_amps_channels else np.empty((0, max(1, max_pulse_events)), dtype=np.float64)

            pulse_prms_serialized = []
            if pulse_prms_channels is not None and len(pulse_prms_channels) > 0:
                for prm_item in pulse_prms_channels:
                    try:
                        pulse_prms_serialized.append(json.dumps(prm_item))
                    except TypeError:
                        pulse_prms_serialized.append(str(prm_item))
                pulse_prms_to_save = np.array(pulse_prms_serialized, dtype=str) # Save as string array
            else:
                pulse_prms_to_save = np.empty((0,), dtype=str)

            # --- Save to Zarr under recording_group/segment_name (NO 'segments' intermediate group) ---
            segment_group_within_recording = recording_group.require_group(segment_name)
            
            chunks_2d = (1, max(1, max_pulse_events))

            zarr.create(
                shape=pulse_times_to_save.shape, 
                dtype=pulse_times_to_save.dtype, 
                chunks=chunks_2d, 
                store=segment_group_within_recording.store, 
                path=f'{segment_group_within_recording.path}/pulse_times', 
                overwrite=True,
                data=pulse_times_to_save 
            )

            zarr.create(
                shape=pulse_amps_to_save.shape, 
                dtype=pulse_amps_to_save.dtype, 
                chunks=chunks_2d, 
                store=segment_group_within_recording.store, 
                path=f'{segment_group_within_recording.path}/pulse_amplitudes', 
                overwrite=True,
                data=pulse_amps_to_save 
            )

            zarr.create(
                shape=pulse_prms_to_save.shape, 
                dtype=pulse_prms_to_save.dtype,
                chunks=(1,) if pulse_prms_to_save.shape[0] > 0 else (1,),
                store=segment_group_within_recording.store, 
                path=f'{segment_group_within_recording.path}/pulse_parameters', 
                overwrite=True,
                data=pulse_prms_to_save
            )

            # --- Audio File Detection and Storage in Zarr Attributes ---
            # Attempt to find a WAV file with the same name as the segment
            # in the same directory as the input ZIP file.
            potential_wav_path = os.path.join(input_item_dir, f"{segment_name}.wav")
            audio_found = False
            if os.path.exists(potential_wav_path):
                # Store the absolute path of the audio file in the segment's Zarr attributes
                segment_group_within_recording.attrs['audio_path'] = potential_wav_path
                print(f"  [INFO] Associated audio file found and path stored for '{segment_name}': {potential_wav_path}")
                audio_found = True
            else:
                segment_group_within_recording.attrs['audio_path'] = None # Explicitly set to None if not found
                # print(f"  [INFO] No direct audio file found for '{segment_name}' at: {potential_wav_path}")

            # --- Plot electrodogram ---
            fig, ax = plt.subplots(figsize=(12, 6))
            
            all_times_flat = np.concatenate(pulse_times_channels) if pulse_times_channels and any(len(t) > 0 for t in pulse_times_channels) else np.array([])
            all_amplitudes_flat = np.concatenate(pulse_amps_channels) if pulse_amps_channels and any(len(a) > 0 for a in pulse_amps_channels) else np.array([])
            
            all_channels_flat = []
            if pulse_times_channels:
                for i, t in enumerate(pulse_times_channels):
                    all_channels_flat.extend([i + 1] * len(t))
            all_channels_flat = np.array(all_channels_flat)

            if all_times_flat.size > 0:
                plot_electrodogram(
                    ax=ax,
                    pulse_times=all_times_flat,
                    pulse_channels=all_channels_flat,
                    pulse_amplitudes=all_amplitudes_flat,
                    fs=fs,
                    title=f"Vectorized Electrodogram: {segment_name}",
                    reverse_channels=(system_type == SYSTEM_COCHLEAR)
                )
            else:
                ax.set_title(f"Vectorized Electrodogram: {segment_name} (No Pulses Found)")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Channel')

            general_plots_dir = os.path.join(parent_output_dir_for_root_zarr, "plots")
            os.makedirs(general_plots_dir, exist_ok=True)
            safe_recording_path_for_filename = full_recording_group_zarr_path.replace('/', '_').replace('\\', '_').replace(':', '')
            plot_filename = os.path.join(general_plots_dir, f"{safe_recording_path_for_filename}_{segment_name}_electrodogram.png")
            fig.savefig(plot_filename)
            plt.close(fig)
            
            # print(f"Finished: {segment_name} for recording '{full_recording_group_zarr_path}'")
            processed_segments_count += 1 # Increment processed segments for this recording

        except Exception as e:
            print(f"[ERROR] Segment '{segment_name}' in recording '{full_recording_group_zarr_path}' failed: {e}")
            print(traceback.format_exc())

    print(f"--- Finished processing {processed_segments_count} segments for recording '{full_recording_group_zarr_path}'. ---")


def main_pipeline(root_dataset_folder, output_zarr_root_path, skip_audio=False):
    """
    Main pipeline function to recursively process CI recording data from a root folder,
    vectorize it, generate electrodograms, and save to a hierarchical Zarr library.

    Args:
        root_dataset_folder (str): Absolute path to the top-level folder of the raw dataset
                                   (e.g., "C:/TESTING" which contains Recordings_Cochlear/AB).
        output_zarr_root_path (str): Path to the ROOT Zarr library directory
                                     (e.g., "C:/output/BSCProject_LIBRARY.zarr").
        skip_audio (bool): Whether to skip audio processing.
    """
    absolute_root_dataset_folder = os.path.abspath(root_dataset_folder)
    absolute_output_zarr_root_path = os.path.abspath(output_zarr_root_path)
    
    print(f"\n--- Starting Full Dataset Processing ---")
    print(f"Scanning root dataset folder: {absolute_root_dataset_folder}")
    print(f"Building Zarr library at: {absolute_output_zarr_root_path}")

    # Ensure the parent directory for the root Zarr store exists
    parent_output_dir_for_root_zarr = os.path.dirname(absolute_output_zarr_root_path)
    if parent_output_dir_for_root_zarr and not os.path.exists(parent_output_dir_for_root_zarr):
        os.makedirs(parent_output_dir_for_root_zarr, exist_ok=True)
        print(f"Created parent output directory for root Zarr store: {parent_output_dir_for_root_zarr}")

    # --- Open/Create the ROOT Zarr store (e.g., BSCProject_LIBRARY.zarr) ---
    try:
        root_zarr_store = zarr.open_group(absolute_output_zarr_root_path, mode='a')
        print(f"Root Zarr store opened/created at: '{absolute_output_zarr_root_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to open/create Root Zarr store at '{absolute_output_zarr_root_path}': {e}")
        print(traceback.format_exc())
        return

    # Find all relevant recording items (ZIP files or unzipped folders with 'rec' dir)
    # This example specifically looks for ZIP files. Adjust if you have unzipped folders.
    recording_items = []
    for dirpath, dirnames, filenames in os.walk(absolute_root_dataset_folder):
        for f in filenames:
            if f.endswith('.zip'):
                recording_items.append(os.path.join(dirpath, f))
        # You could also add logic here to find unzipped directories that contain 'rec' and '__info__'
        # For instance:
        # if 'rec' in dirnames and '__info__' in dirnames:
        #     recording_items.append(dirpath) # Add the directory itself if it's a recording folder
    
    if not recording_items:
        print(f"[WARN] No ZIP files found under '{absolute_root_dataset_folder}'. Nothing to process.")
        return

    print(f"Found {len(recording_items)} recording items (ZIP files) to process.")

    try:
        from tqdm import tqdm
        main_iterator = tqdm(recording_items, desc="Overall Progress")
    except ImportError:
        main_iterator = recording_items
        print("Install 'tqdm' for a progress bar (pip install tqdm).")

    overall_processed_recordings = 0
    for item_path in main_iterator:
        try:
            process_single_recording(item_path, absolute_root_dataset_folder, root_zarr_store, parent_output_dir_for_root_zarr)
            overall_processed_recordings += 1
        except Exception as e:
            print(f"[ERROR] Critical error processing '{item_path}': {e}")
            print(traceback.format_exc())

    print(f"\n--- Full Dataset Processing Complete. Processed {overall_processed_recordings} recordings. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively process CI recording data and generate Zarr library.")
    parser.add_argument('--root_dataset_folder', type=str, required=True,
                        help="Absolute path to the top-level folder of the raw dataset "
                             "(e.g., C:/MyClientData where 'Recordings_Cochlear' and 'Recordings_AB' reside).")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to the ROOT Zarr library directory (e.g., C:/output/BSCProject_LIBRARY.zarr). "
                             "This will be created or appended to.")
    parser.add_argument('--skip_audio', action='store_true',
                        help="Skip audio processing (currently not implemented in this script).")

    args = parser.parse_args()

    # Use Agg backend for matplotlib if not running in an interactive GUI environment
    try:
        plt.get_current_fig_manager().canvas.get_tk_widget()
    except Exception:
        plt.switch_backend('Agg')
        print("[INFO] Using Agg backend for Matplotlib")

    main_pipeline(args.root_dataset_folder, args.output, args.skip_audio)

