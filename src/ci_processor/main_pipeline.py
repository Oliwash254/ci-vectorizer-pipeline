import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import traceback
import zipfile
import numcodecs
import json
import shutil # Import shutil for rmtree

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


def detect_system_type(file_path):
    """
    Detect CI system type from the file path string.
    Expects 'cochlear' or 'ab' to be present in the lowercased path.
    """
    path_lower = str(file_path).lower()
    if "cochlear" in path_lower:
        return SYSTEM_COCHLEAR
    elif "ab" in path_lower:
        return SYSTEM_AB
    else:
        print("[WARN] Could not detect system type from file name. Defaulting to 'AB'.")
        return SYSTEM_AB


def process_data(input_path, output_zarr_root_path, skip_audio=False):
    """
    Main pipeline function to process CI recording data, vectorize it,
    generate electrodograms, and save to a Zarr library.
    
    output_zarr_root_path: The path to the *root* Zarr library (e.g., 'output/ci_library.zarr').
                           A new group for each recording will be created inside this root.
    """
    # Resolve and print absolute paths for clarity
    absolute_input_path = os.path.abspath(input_path)
    absolute_output_zarr_root_path = os.path.abspath(output_zarr_root_path)
    
    print(f"Starting processing for input: {absolute_input_path}")
    print(f"Output Zarr data will be organized under the root: {absolute_output_zarr_root_path}")

    # The root Zarr store needs its parent directory to exist
    parent_output_dir_for_root_zarr = os.path.dirname(absolute_output_zarr_root_path)
    if parent_output_dir_for_root_zarr and not os.path.exists(parent_output_dir_for_root_zarr):
        os.makedirs(parent_output_dir_for_root_zarr, exist_ok=True)
        print(f"Created parent output directory for root Zarr store: {parent_output_dir_for_root_zarr}")

    # --- Open/Create the ROOT Zarr store (e.g., ci_library.zarr) ---
    try:
        # Use open_group(mode='a') to append to existing or create if not exists
        root_zarr_store = zarr.open_group(absolute_output_zarr_root_path, mode='a')
        print(f"Root Zarr store opened/created at: '{absolute_output_zarr_root_path}'")
        if os.path.isdir(absolute_output_zarr_root_path):
            print(f"Confirmed: Root Zarr store directory '{absolute_output_zarr_root_path}' now exists on disk.")
        else:
            print(f"Warning: Root Zarr store directory '{absolute_output_zarr_root_path}' does NOT appear to exist after creation attempt by Zarr. Check permissions or security software.")
    except Exception as e:
        print(f"[ERROR] Failed to open/create Root Zarr store at '{absolute_output_zarr_root_path}': {e}")
        print(traceback.format_exc())
        return

    # --- Extract Recording ID from input_path (e.g., "AB_recording_20250522_2057") ---
    # This ID will be the new intermediate group name
    recording_id = os.path.splitext(os.path.basename(input_path))[0]
    print(f"Extracted Recording ID from input path: '{recording_id}'")

    # --- Create/Access the specific recording group within the root Zarr store ---
    try:
        recording_group = root_zarr_store.require_group(recording_id)
        print(f"Recording group '{recording_id}' created/accessed within root Zarr store.")
    except Exception as e:
        print(f"[ERROR] Failed to create/access recording group '{recording_id}': {e}")
        print(traceback.format_exc())
        return

    dat = None
    try:
        if os.path.isdir(input_path):
            rec_dir = os.path.join(input_path, 'rec')
            info_dir = os.path.join(input_path, '__info__')
            if not os.path.isdir(rec_dir):
                raise ValueError(f"Missing 'rec' folder in directory: {rec_dir}")
            dat = folder_to_npdict(rec_dir, info_dir=info_dir if os.path.isdir(info_dir) else None)
        elif zipfile.is_zipfile(input_path):
            dat = zip_to_npdict(input_path, prefer_unzipped=True)
        else:
            raise ValueError("Input path must be a directory or ZIP file.")
    except Exception as e:
        print(f"[ERROR] Failed to load CI data: {traceback.format_exc()}")
        return

    if not dat or 'rec' not in dat or not dat['rec']:
        print("[ERROR] No 'rec' section found in data. Aborting.")
        return

    try:
        system_type = detect_system_type(input_path)
        print(f"Detected system type: {system_type.upper()}")
        vectorizer = get_vectorizer(system_type)
    except Exception as e:
        print(f"[ERROR] Failed to initialize vectorizer: {traceback.format_exc()}")
        return

    info = dat.get('__info__', {})
    fs = info.get('fs_scope') or info.get('fs')

    if fs is None or fs == 0:
        print("[ERROR] Sampling rate ('fs_scope' or 'fs') not found or zero. Aborting.")
        return

    try:
        from tqdm import tqdm
        segments_iterator = tqdm(dat['rec'].items(), desc="Processing segments")
    except ImportError:
        segments_iterator = dat['rec'].items()
        print("Install 'tqdm' for a progress bar (pip install tqdm).")

    processed = 0

    # Ensure the 'segments' group exists within the recording_group
    recording_segments_group = recording_group.require_group('segments')

    for segment_name, segment_data in segments_iterator:
        try:
            if not isinstance(segment_data, dict) or not all(isinstance(v, np.ndarray) for v in segment_data.values()):
                raise ValueError("Segment data must be a dict of NumPy arrays.")

            keys_sorted = sorted(segment_data.keys(), key=lambda k: int(k) if k.isdigit() else k)
            X_list = [segment_data[k] for k in keys_sorted]
            max_len = max(len(x) for x in X_list)
            X = np.stack([np.pad(x, (0, max_len - len(x))) for x in X_list])

            pulse_times_channels, pulse_amps_channels, pulse_prms_channels = vectorizer.vectorize(X, fs)

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
                pulse_prms_to_save = np.array(pulse_prms_serialized, dtype=str)
            else:
                pulse_prms_to_save = np.empty((0,), dtype=str)

            # --- Save to Zarr under recording_group/segments/segment_name ---
            segment_group_within_recording = recording_segments_group.require_group(segment_name)
            
            chunks_2d = (1, max(1, max_pulse_events))

            zarr.create(
                shape=pulse_times_to_save.shape, 
                dtype=pulse_times_to_save.dtype, 
                chunks=chunks_2d, 
                store=segment_group_within_recording.store, 
                path=f'{segment_group_within_recording.path}/pulse_times', # Use path relative to its group
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

            # Ensure plots are saved to the parent directory of the root Zarr store
            plot_filename = os.path.join(parent_output_dir_for_root_zarr, f"{recording_id}_{segment_name}_electrodogram.png")
            fig.savefig(plot_filename)
            plt.close(fig)
            
            print(f"Finished: {segment_name} for recording '{recording_id}'")
            processed += 1

        except Exception as e:
            print(f"[ERROR] Segment {segment_name} for recording '{recording_id}' failed: {e}")
            print(traceback.format_exc())

    print(f"[INFO] Done. Segments processed: {processed} for recording '{recording_id}'.")
    # --- Final verification after all segments are processed ---
    if os.path.isdir(absolute_output_zarr_root_path):
        print(f"\n--- Final Zarr Store Check ---")
        print(f"The root Zarr store directory '{absolute_output_zarr_root_path}' still exists.")
        
        try:
            # Check for the newly created recording group
            if recording_id in root_zarr_store.keys():
                print(f"Confirmed: Recording group '{recording_id}' exists within the root Zarr store.")
                # Also check for the segments subgroup within it
                if 'segments' in root_zarr_store[recording_id].keys():
                    print(f"Confirmed: 'segments' subgroup exists under '{recording_id}'.")
                    # List segments in the Zarr store for this recording
                    segments_in_zarr = list(root_zarr_store[recording_id]['segments'].keys())
                    if segments_in_zarr:
                        print(f"Found {len(segments_in_zarr)} segments in '{recording_id}/segments'. First segment: '{segments_in_zarr[0]}'.")
                        first_segment_path_in_zarr = os.path.join(recording_id, 'segments', segments_in_zarr[0], 'pulse_times', '.zarray')
                        if root_zarr_store.get(first_segment_path_in_zarr): # Check if the Zarr array exists
                            print(f"Confirmed: A 'pulse_times' array exists for '{segments_in_zarr[0]}' inside '{recording_id}'.")
                        else:
                            print(f"Warning: 'pulse_times' array not found for '{segments_in_zarr[0]}' inside '{recording_id}'. Data might not be written correctly.")
                    else:
                        print(f"Warning: No segments found within '{recording_id}/segments' in the Zarr store.")
                else:
                    print(f"Warning: 'segments' subgroup not found under recording group '{recording_id}'.")
            else:
                print(f"Warning: Recording group '{recording_id}' not found within the root Zarr store.")
        except Exception as e:
            print(f"Error during final Zarr store content check: {e}")
            print(traceback.log_exception(e)) # Use log_exception for full traceback without raising
    else:
        print(f"\n--- Final Zarr Store Check ---")
        print(f"Critical: The root Zarr store directory '{absolute_output_zarr_root_path}' does NOT exist after processing. There's a serious issue with file system writes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI Processor Pipeline")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to CI zip or folder (e.g., C:\\data\\recording.zip)")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to the ROOT Zarr library directory (e.g., C:\\output\\ci_library.zarr). "
                             "A new group for each recording will be created inside this root.")
    parser.add_argument('--skip_audio', action='store_true',
                        help="Skip audio processing (if applicable).")

    args = parser.parse_args()

    try:
        plt.get_current_fig_manager().canvas.get_tk_widget()
    except Exception:
        plt.switch_backend('Agg')
        print("[INFO] Using Agg backend for Matplotlib")

    process_data(args.input_path, args.output, skip_audio=args.skip_audio)

