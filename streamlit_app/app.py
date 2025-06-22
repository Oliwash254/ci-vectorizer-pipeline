import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import tempfile
import traceback
import io
import sys # Import io for byte stream handling
from pathlib import Path
import json
import zarr
src_path = Path(os.getcwd()) / "src"
if src_path.exists():
    sys.path.append(str(src_path))
else:
    print(f"[ERROR] Couldn't find src path at: {src_path}")

# --- IMPORTANT: Troubleshooting "RuntimeError: dictionary changed size during iteration" ---
# This error often occurs due to a race condition in Streamlit's file watcher (watchdog).
# It's usually not a bug in your application code.
#
# Common solutions:
# 1. Upgrade Streamlit and watchdog:
#    pip install --upgrade streamlit watchdog
#
# 2. Disable Streamlit's file watcher (if solution 1 doesn't work, or for deployment):
#    Run your app with:
#    streamlit run app.py --server.fileWatcherType none
#    Or add this to your ~/.streamlit/config.toml file:
#    [server]
#    fileWatcherType = "none"
# --- End Troubleshooting Guide ---





try:
    from ci_processor.ci_vectorization.npdict import zip_to_npdict, folder_to_npdict # Keep folder_to_npdict for completeness
    from ci_processor.ci_vectorization.vectorizers import get_vectorizer
    from ci_processor.electrodogram import plot_electrodogram
    from ci_processor.ci_vectorization.vectorizers import SYSTEM_COCHLEAR # For clarity on system type comparison

except ImportError as e:
    st.error(f"Failed to import ci_processor modules. Please ensure your 'ci_processor' "
             f"package is correctly installed and accessible in your Python environment. "
             f"Error: {e}")
    st.stop()


# --- Function to detect system type ---
def detect_system_type(file_path):
    """
    Detect CI system type from the file path string.
    Expects 'cochlear' or 'ab' to be present in the lowercased path.
    """
    path_lower = str(file_path).lower()
    if "cochlear" in path_lower:
        return "cochlear"
    elif "ab" in path_lower:
        return "ab"
    else:
        st.warning("Could not reliably detect system type from file name/ID. Defaulting to 'ab'. "
                   "For best results, ensure 'cochlear' or 'ab' is in your recording's name.")
        return "ab"


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="CI Electrodogram Visualizer")

st.title("🦻 CI Electrodogram Visualizer")
st.markdown("""
Choose an input method to visualize CI electrodograms, view metadata, and download processed data.
""")

# Initialize session state for storing data and plot data
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'input_method': 'Upload ZIP File', # Default input method
        'zarr_root_path': '',
        'dat': None, # This will be the npdict when loading from ZIP, or None when loading from Zarr
        'zarr_store_obj': None, # The opened root Zarr store object when loading from Zarr
        'recording_ids': [], # List of recording IDs found in the Zarr store
        'selected_recording_id': None,
        'system_type': None,
        'segment_keys': [],
        'selected_segment_key': None,
        'plot_image_bytes': None,
        'audio_file_path': None,
        'download_pulse_times': None,
        'download_pulse_amplitudes': None,
        'download_pulse_parameters': None,
        'temp_dir': None,
        'uploaded_file_name': None
    }

# Input Method Selection
st.session_state.app_state['input_method'] = st.radio(
    "Select Input Method:",
    ['Upload ZIP File', 'Load from Local Zarr Library'],
    index=0 if st.session_state.app_state['input_method'] == 'Upload ZIP File' else 1,
    key='input_method_radio'
)

# --- Logic for "Upload ZIP File" ---
uploaded_file = None
if st.session_state.app_state['input_method'] == 'Upload ZIP File':
    if st.session_state.app_state['dat'] is None or st.session_state.app_state['zarr_store_obj'] is not None:
        # Show uploader if no data loaded OR if previously loaded from Zarr (and now switching back)
        uploaded_file = st.file_uploader("Upload CI Recording ZIP File", type="zip", key="main_file_uploader")
    else:
        # If data is already loaded from ZIP, show a button to clear and upload new
        st.sidebar.markdown("---")
        if st.sidebar.button("Upload New ZIP File", key="clear_upload_button_zip"):
            # Clear all relevant session state to reset the app
            st.session_state.app_state = {k: None if k not in ['input_method', 'zarr_root_path'] else st.session_state.app_state[k] for k in st.session_state.app_state}
            st.session_state.app_state['input_method'] = 'Upload ZIP File' # Ensure correct method is set
            st.rerun()

    if uploaded_file is not None:
        # This block executes when a file is freshly uploaded (not on subsequent reruns of the same file)
        # Check if this is a new upload or rerun of the same file
        if st.session_state.app_state['uploaded_file_name'] != uploaded_file.name or st.session_state.app_state['dat'] is None:
            
            # Clear previous temp_dir if it exists and is not None
            if st.session_state.app_state['temp_dir'] and os.path.exists(st.session_state.app_state['temp_dir']):
                import shutil
                try:
                    shutil.rmtree(st.session_state.app_state['temp_dir'])
                except OSError as e:
                    st.warning(f"Could not remove old temporary directory: {e}. It might be in use.")
                st.session_state.app_state['temp_dir'] = None

            temp_dir_obj = tempfile.TemporaryDirectory()
            st.session_state.app_state['temp_dir'] = temp_dir_obj.name
            st.session_state.app_state['uploaded_file_name'] = uploaded_file.name
            
            zip_path = os.path.join(st.session_state.app_state['temp_dir'], uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File '{uploaded_file.name}' uploaded successfully! Extracting contents...")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(st.session_state.app_state['temp_dir'])
                st.info(f"ZIP contents extracted to: {st.session_state.app_state['temp_dir']}")

                st.session_state.app_state['dat'] = zip_to_npdict(zip_path, prefer_unzipped=True)
                st.session_state.app_state['zarr_store_obj'] = None # Clear Zarr store if switching to ZIP
                st.session_state.app_state['recording_ids'] = [] # Not applicable for direct ZIP load
                st.session_state.app_state['selected_recording_id'] = None
                
                if 'rec' in st.session_state.app_state['dat'] and st.session_state.app_state['dat']['rec']:
                    st.session_state.app_state['segment_keys'] = list(st.session_state.app_state['dat']['rec'].keys())
                    st.session_state.app_state['selected_segment_key'] = st.session_state.app_state['segment_keys'][0] if st.session_state.app_state['segment_keys'] else None
                else:
                    st.session_state.app_state['segment_keys'] = []
                    st.session_state.app_state['selected_segment_key'] = None
                    st.error("No recording segments found in the 'rec' folder of the ZIP file.")
                
                # Reset plot and download data when new file is uploaded
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['download_pulse_times'] = None
                st.session_state.app_state['download_pulse_amplitudes'] = None
                st.session_state.app_state['download_pulse_parameters'] = None
                st.session_state.app_state['audio_file_path'] = None
                
                st.rerun() # Rerun to properly display data after initial upload processing

            except Exception as e:
                st.error(f"An error occurred during file processing or data loading: {e}")
                st.exception(e)
                st.session_state.app_state = { # Reset app state on critical error
                    'input_method': 'Upload ZIP File', 'zarr_root_path': '',
                    'dat': None, 'zarr_store_obj': None, 'recording_ids': [],
                    'selected_recording_id': None, 'system_type': None,
                    'segment_keys': [], 'selected_segment_key': None,
                    'plot_image_bytes': None, 'audio_file_path': None,
                    'download_pulse_times': None, 'download_pulse_amplitudes': None,
                    'download_pulse_parameters': None, 'temp_dir': None,
                    'uploaded_file_name': None
                }
                st.stop()
        else:
            # On rerun of same file, rewind buffer to ensure consistency
            uploaded_file.seek(0)
            
elif st.session_state.app_state['input_method'] == 'Load from Local Zarr Library':
    # Capture path input and strip quotes immediately
    zarr_path_input_raw = st.text_input(
        "Enter Path to Local Zarr Library (e.g., C:/path/to/ci_library.zarr):",
        value=st.session_state.app_state['zarr_root_path'],
        key='zarr_path_input'
    )
    # Strip any potential leading/trailing quotes from the path
    st.session_state.app_state['zarr_root_path'] = zarr_path_input_raw.strip('"')


    if st.session_state.app_state['zarr_root_path']:
        if st.button("Load Zarr Library", key='load_zarr_button'):
            try:
                # Clear previous state from ZIP upload
                st.session_state.app_state['dat'] = None
                st.session_state.app_state['temp_dir'] = None
                st.session_state.app_state['uploaded_file_name'] = None

                # Open the root Zarr store
                root_zarr = zarr.open_group(st.session_state.app_state['zarr_root_path'], mode='r')
                st.session_state.app_state['zarr_store_obj'] = root_zarr
                
                # List all top-level groups (these are your recording IDs)
                # FIX: Changed zarr.hierarchy.Group to zarr.Group
                recording_ids = [key for key in root_zarr.keys() if isinstance(root_zarr[key], zarr.Group)]
                st.session_state.app_state['recording_ids'] = recording_ids
                
                if recording_ids:
                    # Set initial selected recording ID
                    st.session_state.app_state['selected_recording_id'] = recording_ids[0]
                    # System type can be detected from the recording ID name, or we might need metadata inside Zarr
                    st.session_state.app_state['system_type'] = detect_system_type(recording_ids[0])
                else:
                    st.warning("No recording groups found in the specified Zarr library. Ensure structure is `root/recording_id/segments/`.")
                    st.session_state.app_state['selected_recording_id'] = None
                    st.session_state.app_state['recording_ids'] = []
                
                # Clear segment keys until a recording is selected
                st.session_state.app_state['segment_keys'] = []
                st.session_state.app_state['selected_segment_key'] = None
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['audio_file_path'] = None
                st.session_state.app_state['download_pulse_times'] = None
                st.session_state.app_state['download_pulse_amplitudes'] = None
                st.session_state.app_state['download_pulse_parameters'] = None

                st.success(f"Zarr library loaded from {st.session_state.app_state['zarr_root_path']}. Found {len(recording_ids)} recordings.")
                st.rerun() # Rerun to display recording selector
            except Exception as e:
                st.error(f"Failed to load Zarr library from {st.session_state.app_state['zarr_root_path']}: {e}")
                st.exception(e)
                st.session_state.app_state['zarr_store_obj'] = None
                st.session_state.app_state['recording_ids'] = []
                st.session_state.app_state['selected_recording_id'] = None

    # --- Display Content if Data is Loaded (either from ZIP or Zarr) ---
    if st.session_state.app_state['dat'] or st.session_state.app_state['zarr_store_obj']:
        st.sidebar.header("Data & Analysis Options")
        if st.session_state.app_state['system_type']: # Only display if system_type is determined
            st.sidebar.write(f"Detected System Type: **{st.session_state.app_state['system_type'].upper()}**")

        # Metadata Section
        if st.session_state.app_state['input_method'] == 'Upload ZIP File' and st.session_state.app_state['dat'] and st.session_state.app_state['dat'].get('__info__'):
            with st.sidebar.expander("View Metadata (__info__)"):
                st.json(st.session_state.app_state['dat']['__info__'])
        elif st.session_state.app_state['input_method'] == 'Load from Local Zarr Library' and st.session_state.app_state['zarr_store_obj'] and st.session_state.app_state['selected_recording_id']:
            with st.sidebar.expander(f"View Recording Metadata ({st.session_state.app_state['selected_recording_id']})"):
                current_recording_group = st.session_state.app_state['zarr_store_obj'].get(st.session_state.app_state['selected_recording_id'])
                if current_recording_group and '__info__' in current_recording_group:
                    try:
                        info_zarr_array = current_recording_group['__info__'][:]
                        if info_zarr_array.dtype.kind in ('U', 'S') and info_zarr_array.size > 0:
                            try:
                                deserialized_info = json.loads(info_zarr_array.item())
                                st.json(deserialized_info)
                            except (json.JSONDecodeError, ValueError) as json_err:
                                st.warning(f"Could not decode __info__ JSON: {json_err}. Displaying raw data.")
                                st.write(info_zarr_array.item())
                        else:
                            st.write("Raw Zarr __info__ data (not JSON string):")
                            st.write(info_zarr_array)
                    except Exception as info_e:
                        st.warning(f"Could not load __info__ from Zarr: {info_e}. Ensure your main_pipeline.py saves it correctly.")
                        st.write("No direct '__info__' array found or could not be loaded for this recording in Zarr.")
                else:
                    st.info("No '__info__' array found for this recording in Zarr.")
        else:
            st.info("No metadata available or selected.")


        # --- Recording ID Selection (if loading from Zarr) ---
        current_recording_group = None
        if st.session_state.app_state['input_method'] == 'Load from Local Zarr Library' and st.session_state.app_state['recording_ids']:
            selected_recording_id_from_ui = st.sidebar.selectbox(
                "Select a Recording ID:",
                options=st.session_state.app_state['recording_ids'],
                index=st.session_state.app_state['recording_ids'].index(st.session_state.app_state['selected_recording_id']) if st.session_state.app_state['selected_recording_id'] in st.session_state.app_state['recording_ids'] else 0,
                key='recording_id_select'
            )

            if selected_recording_id_from_ui != st.session_state.app_state['selected_recording_id']:
                st.session_state.app_state['selected_recording_id'] = selected_recording_id_from_ui
                st.session_state.app_state['system_type'] = detect_system_type(selected_recording_id_from_ui) 
                st.session_state.app_state['segment_keys'] = []
                st.session_state.app_state['selected_segment_key'] = None
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['audio_file_path'] = None
                st.session_state.app_state['download_pulse_times'] = None
                st.session_state.app_state['download_pulse_amplitudes'] = None
                st.session_state.app_state['download_pulse_parameters'] = None
                st.rerun()

            current_recording_group = st.session_state.app_state['zarr_store_obj'].get(st.session_state.app_state['selected_recording_id'])
            if current_recording_group and 'segments' in current_recording_group:
                st.session_state.app_state['segment_keys'] = list(current_recording_group['segments'].keys())
                if st.session_state.app_state['selected_segment_key'] not in st.session_state.app_state['segment_keys']:
                    st.session_state.app_state['selected_segment_key'] = st.session_state.app_state['segment_keys'][0] if st.session_state.app_state['segment_keys'] else None
            else:
                st.warning(f"No 'segments' group found in recording '{st.session_state.app_state['selected_recording_id']}'. Check Zarr structure.")
                st.session_state.app_state['segment_keys'] = []
                st.session_state.app_state['selected_segment_key'] = None


        # --- Segment Selection (common to both input methods) ---
        if st.session_state.app_state['segment_keys']:
            current_selected_index = 0
            if st.session_state.app_state['selected_segment_key'] in st.session_state.app_state['segment_keys']:
                 current_selected_index = st.session_state.app_state['segment_keys'].index(st.session_state.app_state['selected_segment_key'])

            selected_segment_key_from_ui = st.sidebar.selectbox(
                "Select a Recording Segment:",
                options=st.session_state.app_state['segment_keys'],
                index=current_selected_index,
                key="segment_select"
            )
            
            if selected_segment_key_from_ui != st.session_state.app_state['selected_segment_key']:
                st.session_state.app_state['selected_segment_key'] = selected_segment_key_from_ui
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['download_pulse_times'] = None
                st.session_state.app_state['download_pulse_amplitudes'] = None
                st.session_state.app_state['download_pulse_parameters'] = None
                st.session_state.app_state['audio_file_path'] = None
                st.rerun()

            # --- Process and Display Selected Segment ---
            if st.session_state.app_state['selected_segment_key']:
                current_segment_name = st.session_state.app_state['selected_segment_key']
                st.subheader(f"Segment: `{current_segment_name}`")

                pulse_times = []
                pulse_amplitudes = []
                pulse_prms = []
                fs = None

                if st.session_state.app_state['input_method'] == 'Upload ZIP File' and st.session_state.app_state['dat']:
                    segment_data = st.session_state.app_state['dat']['rec'][current_segment_name]
                    fs = st.session_state.app_state['dat'].get('__info__', {}).get('fs_scope') or st.session_state.app_state['dat'].get('__info__', {}).get('fs')
                    
                    extracted_data_root = os.path.join(st.session_state.app_state['temp_dir'], os.path.splitext(st.session_state.app_state['uploaded_file_name'])[0])
                    potential_wav_file = os.path.join(extracted_data_root, f"{current_segment_name}.wav")
                    if os.path.exists(potential_wav_file):
                        st.session_state.app_state['audio_file_path'] = potential_wav_file
                    else:
                        potential_wav_file_flat = os.path.join(st.session_state.app_state['temp_dir'], f"{current_segment_name}.wav")
                        if os.path.exists(potential_wav_file_flat):
                            st.session_state.app_state['audio_file_path'] = potential_wav_file_flat
                        else:
                            st.session_state.app_state['audio_file_path'] = None

                    with st.spinner(f"Vectorizing data from ZIP for '{current_segment_name}'..."):
                        sorted_channel_keys = sorted(segment_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
                        X_list = [segment_data[ch] for ch in sorted_channel_keys]
                        max_len_X = max(len(arr) for arr in X_list)
                        X_padded = [np.pad(arr, (0, max_len_X - len(arr))) for arr in X_list]
                        X = np.array(X_padded)
                        
                        vectorizer_instance = get_vectorizer(st.session_state.app_state['system_type'])
                        pulse_times, pulse_amplitudes, pulse_prms = vectorizer_instance.vectorize(X, fs)


                elif st.session_state.app_state['input_method'] == 'Load from Local Zarr Library' and current_recording_group:
                    zarr_segment_group = current_recording_group['segments'].get(current_segment_name)
                    if zarr_segment_group:
                        pulse_times_channels_zarr = zarr_segment_group.get('pulse_times')
                        pulse_amps_channels_zarr = zarr_segment_group.get('pulse_amplitudes')
                        pulse_prms_channels_zarr = zarr_segment_group.get('pulse_parameters')

                        if pulse_times_channels_zarr is None or pulse_amps_channels_zarr is None or pulse_prms_channels_zarr is None:
                            st.error(f"Missing one or more Zarr arrays (pulse_times, pulse_amplitudes, pulse_parameters) for segment '{current_segment_name}'.")
                            st.stop()

                        pulse_times = [arr[~np.isnan(arr)] for arr in pulse_times_channels_zarr]
                        pulse_amplitudes = [arr[~np.isnan(arr)] for arr in pulse_amps_channels_zarr]
                        
                        pulse_prms = []
                        for serialized_channel_prms_array in pulse_prms_channels_zarr:
                            deserialized_channel_prms = []
                            # Attempt to load each element as JSON. If not JSON, keep as is.
                            if isinstance(serialized_channel_prms_array, str):
                                try:
                                    deserialized_item = json.loads(serialized_channel_prms_array)
                                    deserialized_channel_prms.append(deserialized_item)
                                except (json.JSONDecodeError, ValueError):
                                    deserialized_channel_prms.append(serialized_channel_prms_array)
                            else:
                                 deserialized_channel_prms.append(serialized_channel_prms_array)
                            pulse_prms.append(deserialized_channel_prms)

                        fs = 44100 

                        st.session_state.app_state['audio_file_path'] = None

                    else:
                        st.error(f"Segment '{current_segment_name}' not found in the selected recording's segments group.")
                        st.stop()
                else:
                    st.error("Internal Error: No data source selected or available.")
                    st.stop()


                if fs is None or fs == 0:
                    st.error("Sampling rate ('fs_scope' or 'fs') could not be determined. Cannot proceed with plotting/vectorization.")
                    st.stop()

                # Audio Playback
                if st.session_state.app_state['audio_file_path']:
                    st.write("#### Audio Playback")
                    st.audio(st.session_state.app_state['audio_file_path'], format='audio/wav')
                else:
                    st.info("No audio file found or supported for this segment/source type.")

                # Electrodogram Visualization
                if st.session_state.app_state['plot_image_bytes'] is None:
                    
                    if not pulse_times or not pulse_amplitudes:
                        st.error("Pulse data could not be retrieved for plotting.")
                        st.stop()

                    fig, ax = plt.subplots(figsize=(12, 6))

                    all_times_flat = np.concatenate(pulse_times) if pulse_times and any(len(t) > 0 for t in pulse_times) else np.array([])
                    all_amplitudes_flat = np.concatenate(pulse_amplitudes) if pulse_amplitudes and any(len(a) > 0 for a in pulse_amplitudes) else np.array([])
                    
                    all_channels_flat = []
                    if pulse_times:
                        for i, t in enumerate(pulse_times):
                            all_channels_flat.extend([i + 1] * len(t))
                    all_channels_flat = np.array(all_channels_flat)

                    if all_times_flat.size > 0:
                        plot_electrodogram(
                            ax=ax,
                            pulse_times=all_times_flat,
                            pulse_channels=all_channels_flat,
                            pulse_amplitudes=all_amplitudes_flat,
                            fs=fs,
                            title=f"{current_segment_name} ({st.session_state.app_state['system_type'].replace('ab', 'Advanced Bionics').replace('cochlear', 'Cochlear')}): Vectorized Electrodogram",
                            reverse_channels=(st.session_state.app_state['system_type'] == "cochlear")
                        )
                    else:
                        ax.set_title(f"Vectorized Electrodogram: {current_segment_name} (No Pulses Found)")
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Channel')

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    st.session_state.app_state['plot_image_bytes'] = buf.getvalue()
                    plt.close(fig)

                    st.session_state.app_state['download_pulse_times'] = pulse_times
                    st.session_state.app_state['download_pulse_amplitudes'] = pulse_amplitudes
                    st.session_state.app_state['download_pulse_parameters'] = pulse_prms
                    
                    st.image(st.session_state.app_state['plot_image_bytes'])
                    st.success("Electrodogram generated successfully!")
                else:
                    st.image(st.session_state.app_state['plot_image_bytes'])
                    st.info("Electrodogram loaded from cache.")
                
                st.write("#### Download Processed Data")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.session_state.app_state['plot_image_bytes']:
                        st.download_button(
                            label="Download Electrodogram (PNG)",
                            data=st.session_state.app_state['plot_image_bytes'],
                            file_name=f"{current_segment_name}_electrodogram.png",
                            mime="image/png",
                            key=f"download_plot_{current_segment_name}"
                        )
                    else:
                        st.info("Electrodogram not yet generated.")

                with col2:
                    if st.session_state.app_state['download_pulse_times'] is not None and len(st.session_state.app_state['download_pulse_times']) > 0:
                        current_max_pulse_len = 0
                        if any(len(arr) > 0 for arr in st.session_state.app_state['download_pulse_times']):
                            current_max_pulse_len = max(len(arr) for arr in st.session_state.app_state['download_pulse_times'])

                        times_arr_to_download = np.array([np.pad(arr, (0, current_max_pulse_len - len(arr)), 'constant', constant_values=np.nan) for arr in st.session_state.app_state['download_pulse_times']])
                        
                        times_buf = io.BytesIO()
                        np.save(times_buf, times_arr_to_download)
                        times_buf.seek(0)
                        st.download_button(
                            label="Download Pulse Times (.npy)",
                            data=times_buf,
                            file_name=f"{current_segment_name}_pulse_times.npy",
                            mime="application/octet-stream",
                            key=f"download_times_{current_segment_name}"
                        )
                    else:
                        st.info("Pulse times not available.")

                with col3:
                    if st.session_state.app_state['download_pulse_amplitudes'] is not None and len(st.session_state.app_state['download_pulse_amplitudes']) > 0:
                        current_max_pulse_len = 0
                        if any(len(arr) > 0 for arr in st.session_state.app_state['download_pulse_amplitudes']):
                            current_max_pulse_len = max(len(arr) for arr in st.session_state.app_state['download_pulse_amplitudes'])
                        
                        amps_arr_to_download = np.array([np.pad(arr, (0, current_max_pulse_len - len(arr)), 'constant', constant_values=np.nan) for arr in st.session_state.app_state['download_pulse_amplitudes']])
                        
                        amps_buf = io.BytesIO()
                        np.save(amps_buf, amps_arr_to_download)
                        amps_buf.seek(0)
                        st.download_button(
                            label="Download Pulse Amplitudes (.npy)",
                            data=amps_buf,
                            file_name=f"{current_segment_name}_pulse_amplitudes.npy",
                            mime="application/octet-stream",
                            key=f"download_amps_{current_segment_name}"
                        )
                    else:
                        st.info("Pulse amplitudes not available.")
                
                with col4:
                    if st.session_state.app_state['download_pulse_parameters'] is not None:
                        if any(st.session_state.app_state['download_pulse_parameters']):
                            all_prms_flat = []
                            for channel_prms in st.session_state.app_state['download_pulse_parameters']:
                                if isinstance(channel_prms, list):
                                    all_prms_flat.extend(channel_prms)
                                else:
                                    all_prms_flat.append(channel_prms)

                            prms_json_str = json.dumps(all_prms_flat, indent=4, default=str)
                            
                            prms_buf = io.BytesIO(prms_json_str.encode('utf-8'))
                            st.download_button(
                                label="Download Pulse Parameters (.json)",
                                data=prms_buf,
                                file_name=f"{current_segment_name}_pulse_parameters.json",
                                mime="application/json",
                                key=f"download_prms_{current_segment_name}"
                            )
                        else:
                            st.info("Pulse parameters not available.")
                    else:
                        st.info("Pulse parameters not available.")
        else:
            st.info("Please select a recording and segment to begin processing and visualization.")
    else:
        st.info("Select an input method above and provide a ZIP file or Zarr library path to begin.")

