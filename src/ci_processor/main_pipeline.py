import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import traceback
import zipfile
import numcodecs # Import numcodecs for object_codec (though we'll use it differently)
import json # Import json for serialization

# Import functions from your ci_processor library
# Ensure your ci_processor package is installed in editable mode:
# pip install -e C:\CI_library_project\
try:
    from ci_processor.ci_vectorization.npdict import zip_to_npdict, folder_to_npdict
    from ci_processor.ci_vectorization.vectorizers import get_vectorizer, SYSTEM_AB, SYSTEM_COCHLEAR
    from ci_processor.electrodogram import plot_electrodogram # Ensure this is imported

except ImportError as e:
    print(f"[ERROR] Failed to import ci_processor modules. Please ensure your 'ci_processor' "
          f"package is correctly installed and accessible in your Python environment. "
          f"Error: {e}")
    # Exit if core modules can't be imported, as the pipeline cannot proceed
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


def process_data(input_path, output_path, skip_audio=False):
    """
    Main pipeline function to process CI recording data, vectorize it,
    generate electrodograms, and save to a Zarr library.
    """
    print(f"Starting processing for: {input_path}")
    print(f"Output Zarr will be saved to: {output_path}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Use open_group(mode='a') which creates if it doesn't exist, and appends if it does.
        root = zarr.open_group(output_path, mode='a')
        print(f"Zarr store opened at: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to open Zarr store: {e}")
        return

    dat = None # Initialize dat
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

    for segment_name, segment_data in segments_iterator: # Use segments_iterator
        try:
            if not isinstance(segment_data, dict) or not all(isinstance(v, np.ndarray) for v in segment_data.values()):
                raise ValueError("Segment data must be a dict of NumPy arrays.")

            keys_sorted = sorted(segment_data.keys(), key=lambda k: int(k) if k.isdigit() else k)
            X_list = [segment_data[k] for k in keys_sorted]
            max_len = max(len(x) for x in X_list)
            X = np.stack([np.pad(x, (0, max_len - len(x))) for x in X_list])

            # pulse_times, pulse_amps, pulse_prms are lists of arrays (one per channel)
            pulse_times_channels, pulse_amps_channels, pulse_prms_channels = vectorizer.vectorize(X, fs)

            # --- Prepare data for Zarr saving ---
            max_pulse_events = 0
            if pulse_times_channels and any(len(arr) > 0 for arr in pulse_times_channels):
                max_pulse_events = max(len(arr) for arr in pulse_times_channels)

            pulse_times_to_save = np.array([
                np.pad(arr, (0, max_pulse_events - len(arr)), 'constant', constant_values=np.nan)
                for arr in pulse_times_channels
            ]) if pulse_times_channels else np.empty((0, max(1, max_pulse_events)), dtype=np.float64) # Ensure at least 1 column for empty array

            pulse_amps_to_save = np.array([
                np.pad(arr, (0, max_pulse_events - len(arr)), 'constant', constant_values=np.nan)
                for arr in pulse_amps_channels
            ]) if pulse_amps_channels else np.empty((0, max(1, max_pulse_events)), dtype=np.float64) # Ensure at least 1 column for empty array

            # Serialize pulse_prms_channels to JSON strings for Zarr, as direct object arrays are problematic with your Zarr version
            pulse_prms_serialized = []
            if pulse_prms_channels is not None and len(pulse_prms_channels) > 0:
                for prm_item in pulse_prms_channels:
                    try:
                        # Attempt to serialize to JSON. If it fails, repr() or str() as fallback.
                        # It's crucial that prm_item is JSON serializable if we want to use json.dumps
                        pulse_prms_serialized.append(json.dumps(prm_item))
                    except TypeError:
                        # Fallback to string representation if not JSON serializable
                        pulse_prms_serialized.append(str(prm_item))
                pulse_prms_to_save = np.array(pulse_prms_serialized, dtype=str) # Store as string array
            else:
                pulse_prms_to_save = np.empty((0,), dtype=str) # Empty string array if no data


            # --- Save to Zarr ---
            segment_group = root.require_group(f'segments/{segment_name}')
            
            # Define common chunks tuple for 2D arrays
            chunks_2d = (1, max(1, max_pulse_events)) # Chunk each channel individually

            # Save pulse_times
            zarr.create(
                shape=pulse_times_to_save.shape, 
                dtype=pulse_times_to_save.dtype, 
                chunks=chunks_2d, 
                store=segment_group.store, 
                path=f'{segment_group.path}/pulse_times', 
                overwrite=True,
                data=pulse_times_to_save 
            )

            # Save pulse_amplitudes
            zarr.create(
                shape=pulse_amps_to_save.shape, 
                dtype=pulse_amps_to_save.dtype, 
                chunks=chunks_2d, 
                store=segment_group.store, 
                path=f'{segment_group.path}/pulse_amplitudes', 
                overwrite=True,
                data=pulse_amps_to_save 
            )

            # Save pulse_parameters with string dtype (since we serialized them)
            # No object_codec needed as it's now string data
            zarr.create(
                shape=pulse_prms_to_save.shape, 
                dtype=pulse_prms_to_save.dtype, # This will be 'str'
                chunks=(1,) if pulse_prms_to_save.shape[0] > 0 else (1,), # Chunks for 1D string array
                store=segment_group.store, 
                path=f'{segment_group.path}/pulse_parameters', 
                overwrite=True,
                data=pulse_prms_to_save 
            )

            # --- Plot electrodogram ---
            fig, ax = plt.subplots(figsize=(12, 6))
            
            all_times_flat = np.concatenate(pulse_times_channels) if pulse_times_channels and any(len(t) > 0 for t in pulse_times_channels) else np.array([])
            all_amps_flat = np.concatenate(pulse_amps_channels) if pulse_amps_channels and any(len(a) > 0 for a in pulse_amps_channels) else np.array([])
            
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
                    pulse_amplitudes=all_amps_flat,
                    fs=fs,
                    title=f"Vectorized Electrodogram: {segment_name}",
                    reverse_channels=(system_type == SYSTEM_COCHLEAR)
                )
            else:
                ax.set_title(f"Vectorized Electrodogram: {segment_name} (No Pulses Found)")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Channel')

            plot_filename = os.path.join(output_dir, f"{segment_name}_electrodogram.png")
            fig.savefig(plot_filename)
            plt.close(fig)
            
            print(f"Finished: {segment_name}")
            processed += 1

        except Exception as e:
            print(f"[ERROR] Segment {segment_name} failed: {e}")
            print(traceback.format_exc())

    print(f"[INFO] Done. Segments processed: {processed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI Processor Pipeline")
    parser.add_argument('--input_path', type=str, required=True, help="Path to CI zip or folder")
    parser.add_argument('--output', type=str, required=True, help="Output Zarr path")
    parser.add_argument('--skip_audio', action='store_true', help="Skip audio processing")

    args = parser.parse_args()

    try:
        plt.get_current_fig_manager().canvas.get_tk_widget()
    except Exception:
        plt.switch_backend('Agg')
        print("[INFO] Using Agg backend for Matplotlib")

    process_data(args.input_path, args.output, skip_audio=args.skip_audio)

