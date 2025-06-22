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


# Import functions from your ci_processor library
# This assumes ci_processor is correctly installed in your venv (pip install -e .)
try:
    from ci_processor.ci_vectorization.npdict import zip_to_npdict
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
    file_path_lower = str(file_path).lower()
    if "cochlear" in file_path_lower:
        return "cochlear"
    elif "ab" in file_path_lower:
        return "ab"
    else:
        st.warning("Could not reliably detect system type from file name. Defaulting to 'ab'. "
                   "Please ensure 'cochlear' or 'ab' is in your ZIP file name for auto-detection.")
        return "ab"


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="CI Electrodogram Visualizer")

st.title("🦻 CI Electrodogram Visualizer")
st.markdown("""
Upload your CI recording `.zip` file (containing `rec` and `__info__` folders, and optionally `.wav` files)
to visualize electrodograms, view metadata, and download processed data.
""")

# Initialize session state for storing data and plot data
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'dat': None,
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

uploaded_file = None # Initialize uploaded_file to None by default

# Conditional rendering of file_uploader
if st.session_state.app_state['dat'] is None:
    # Display the file uploader only if no data has been loaded yet
    uploaded_file = st.file_uploader("Upload CI Recording ZIP File", type="zip", key="main_file_uploader")
else:
    # If data is already loaded, show a button to clear and upload a new file
    st.sidebar.markdown("---")
    if st.sidebar.button("Upload New ZIP File", key="clear_upload_button"):
        # Clear all relevant session state to reset the app
        st.session_state.app_state = {
            'dat': None, 'system_type': None, 'segment_keys': [],
            'selected_segment_key': None, 'plot_image_bytes': None,
            'audio_file_path': None, 'download_pulse_times': None,
            'download_pulse_amplitudes': None, 'download_pulse_parameters': None,
            'temp_dir': None, 'uploaded_file_name': None
        }
        st.rerun() # Rerun to show the file uploader again

# Handle file upload logic (this part remains similar, but now depends on the conditionally rendered uploaded_file)
if uploaded_file is not None:
    # This block now only executes when a file is freshly uploaded
    # It will not execute on subsequent reruns unless 'dat' is cleared and a new file is chosen.

    # Ensure temp_dir is managed properly
    if st.session_state.app_state['temp_dir'] and os.path.exists(st.session_state.app_state['temp_dir']):
        import shutil
        try:
            shutil.rmtree(st.session_state.app_state['temp_dir'])
            st.session_state.app_state['temp_dir'] = None
        except OSError as e:
            st.warning(f"Could not remove old temporary directory: {e}. It might be in use.")

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
        st.session_state.app_state['system_type'] = detect_system_type(uploaded_file.name)

        if 'rec' in st.session_state.app_state['dat'] and st.session_state.app_state['dat']['rec']:
            st.session_state.app_state['segment_keys'] = list(st.session_state.app_state['dat']['rec'].keys())
            st.session_state.app_state['selected_segment_key'] = st.session_state.app_state['segment_keys'][0] if st.session_state.app_state['segment_keys'] else None
        else:
            st.session_state.app_state['segment_keys'] = []
            st.session_state.app_state['selected_segment_key'] = None
            st.error("No recording segments found in the 'rec' folder of the ZIP file.")
        
        # Ensure plot/download data is clear for new upload
        st.session_state.app_state['plot_image_bytes'] = None
        st.session_state.app_state['download_pulse_times'] = None
        st.session_state.app_state['download_pulse_amplitudes'] = None
        st.session_state.app_state['download_pulse_parameters'] = None
        st.session_state.app_state['audio_file_path'] = None
        
        st.rerun() # Rerun to properly display data after initial upload processing

    except Exception as e:
        st.error(f"An error occurred during file processing or data loading: {e}")
        st.exception(e)
        st.session_state.app_state.update({ 
            'dat': None, 'system_type': None, 'segment_keys': [],
            'selected_segment_key': None, 'plot_image_bytes': None,
            'audio_file_path': None, 'download_pulse_times': None,
            'download_pulse_amplitudes': None, 'download_pulse_parameters': None,
            'temp_dir': None, 'uploaded_file_name': None
        })
        st.stop()


# --- Display Content if Data is Loaded ---
# This block executes if 'dat' is already populated in session_state (after initial upload processing or on rerun)
if st.session_state.app_state['dat']:
    st.sidebar.header("Data & Analysis Options")
    st.sidebar.write(f"Detected System Type: **{st.session_state.app_state['system_type'].upper()}**")

    # Metadata Section
    with st.sidebar.expander("View Metadata (__info__)"):
        info_data = st.session_state.app_state['dat'].get('__info__', {})
        if info_data:
            st.json(info_data)
        else:
            st.write("No '__info__' section found in the uploaded data.")

    # Segment Selection
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
        
        # Logic to update session state and clear cache if segment selection changes
        if selected_segment_key_from_ui != st.session_state.app_state['selected_segment_key']:
            st.session_state.app_state['selected_segment_key'] = selected_segment_key_from_ui
            st.session_state.app_state['plot_image_bytes'] = None # Clear plot cache to force regeneration
            st.session_state.app_state['download_pulse_times'] = None
            st.session_state.app_state['download_pulse_amplitudes'] = None
            st.session_state.app_state['download_pulse_parameters'] = None
            st.session_state.app_state['audio_file_path'] = None # Clear audio path for new segment
            st.rerun() # Rerun to apply segment change and clear cache properly

        # --- Process and Display Selected Segment ---
        if st.session_state.app_state['selected_segment_key']:
            current_segment_name = st.session_state.app_state['selected_segment_key']
            st.subheader(f"Segment: `{current_segment_name}`")

            # Try to infer WAV file path for audio playback
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
                    st.warning(f"Could not find audio file for segment '{current_segment_name}'. Looked in '{potential_wav_file}' and '{potential_wav_file_flat}'.")

            # Audio Playback
            if st.session_state.app_state['audio_file_path']:
                st.write("#### Audio Playback")
                st.audio(st.session_state.app_state['audio_file_path'], format='audio/wav')
            else:
                st.info("No audio file found for this segment.")

            st.write("#### Electrodogram Visualization")
            # Process and Plot Electrodogram (only if not already cached)
            if st.session_state.app_state['plot_image_bytes'] is None:
                
                segment_data = st.session_state.app_state['dat']['rec'][current_segment_name]
                if not isinstance(segment_data, dict) or not all(isinstance(v, np.ndarray) for v in segment_data.values()):
                    st.error(f"Segment data for '{current_segment_name}' is not in the expected format (dictionary of numpy arrays).")
                    st.stop()
                
                sorted_channel_keys = sorted(segment_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
                X_list = [segment_data[ch] for ch in sorted_channel_keys]
                
                max_len = max(len(arr) for arr in X_list)
                X_padded = [np.pad(arr, (0, max_len - len(arr))) for arr in X_list]
                X = np.array(X_padded)

                fs = st.session_state.app_state['dat'].get('__info__', {}).get('fs_scope')
                if fs is None:
                    fs = st.session_state.app_state['dat'].get('__info__', {}).get('fs')
                
                if fs is None:
                    st.error("Sampling rate ('fs_scope' or 'fs') could not be determined. Cannot proceed.")
                    st.stop()

                with st.spinner(f"Vectorizing and plotting electrodogram for '{current_segment_name}'..."):
                    vectorizer_instance = get_vectorizer(st.session_state.app_state['system_type'])
                    pulse_times, pulse_amplitudes, pulse_prms = vectorizer_instance.vectorize(X, fs)

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

                    # Save figure to bytes for display and download
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    st.session_state.app_state['plot_image_bytes'] = buf.getvalue()
                    plt.close(fig) # Close the figure to free memory

                    # Store data arrays for download
                    st.session_state.app_state['download_pulse_times'] = pulse_times # Store as list of arrays
                    st.session_state.app_state['download_pulse_amplitudes'] = pulse_amplitudes # Store as list of arrays
                    st.session_state.app_state['download_pulse_parameters'] = pulse_prms # Store as list of objects/dicts
                
                st.image(st.session_state.app_state['plot_image_bytes']) # Display the generated image
                st.success("Electrodogram generated successfully!")
            else:
                # If cached, just display it
                st.image(st.session_state.app_state['plot_image_bytes'])
                st.info("Electrodogram loaded from cache.")
            
            # Display Download Buttons for Electrodogram and Data Arrays
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
                if st.session_state.app_state['download_pulse_times'] is not None:
                    # Pad to the max length of any individual channel's pulses
                    current_max_pulse_len = 0
                    if st.session_state.app_state['download_pulse_times'] and any(len(arr) > 0 for arr in st.session_state.app_state['download_pulse_times']):
                        current_max_pulse_len = max(len(arr) for arr in st.session_state.app_state['download_pulse_times'])

                    times_arr_to_download = np.array([np.pad(arr, (0, current_max_pulse_len - len(arr)), 'constant', constant_values=np.nan) for arr in st.session_state.app_state['download_pulse_times']])
                    
                    times_buf = io.BytesIO()
                    np.save(times_buf, times_arr_to_download)
                    times_buf.seek(0)
                    st.download_button(
                        label="Download Pulse Times (.npy)",
                        data=times_buf,
                        file_name=f"{current_segment_name}_pulse_times.npy",
                        mime="application/octet-stream", # Standard mime type for .npy
                        key=f"download_times_{current_segment_name}"
                    )
                else:
                    st.info("Pulse times not available.")

            with col3:
                if st.session_state.app_state['download_pulse_amplitudes'] is not None:
                    current_max_pulse_len = 0
                    if st.session_state.app_state['download_pulse_amplitudes'] and any(len(arr) > 0 for arr in st.session_state.app_state['download_pulse_amplitudes']):
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
        st.info("Please select a segment from the sidebar to begin processing and visualization.")
else:
    # This message is displayed when no file is uploaded yet (initial state)
    st.info("Upload a ZIP file to begin processing and visualization.")

