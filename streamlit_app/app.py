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
    from ci_processor.ci_vectorization.npdict import zip_to_npdict, folder_to_npdict
    from ci_processor.ci_vectorization.vectorizers import get_vectorizer, SYSTEM_AB, SYSTEM_COCHLEAR
    from ci_processor.electrodogram import plot_electrodogram # Ensure this is imported directly

except ImportError as e:
    st.error(f"Failed to import ci_processor modules. Please ensure your 'ci_processor' "
             f"package is correctly installed and accessible in your Python environment. "
             f"Error: {e}")
    st.stop()


# --- Function to detect system type (copied from main_pipeline.py) ---
def detect_system_type(path_or_id):
    """
    Detect CI system type from a path string or recording ID.
    Expects 'cochlear' or 'ab' to be present in the lowercased path/ID.
    """
    path_or_id_lower = str(path_or_id).lower()
    if "cochlear" in path_or_id_lower:
        return SYSTEM_COCHLEAR # Use the constant directly
    elif "ab" in path_or_id_lower:
        return SYSTEM_AB # Use the constant directly
    else:
        # For Zarr-loaded data, if system type isn't directly in ID,
        # it should ideally be stored in Zarr metadata for precise detection.
        # Fallback to default, but warn.
        st.warning(f"Could not reliably detect system type from '{path_or_id}'. Defaulting to 'ab'. "
                   "For best results, ensure 'cochlear' or 'ab' is in your recording's name or Zarr metadata.")
        return SYSTEM_AB # Default to 'ab' if not found


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="CI Electrodogram Visualizer")

st.title("🦻 CI Electrodogram Visualizer")
st.markdown("""
Choose an input method to visualize CI electrodograms, view metadata, and download processed data.
""")

# Initialize session state for storing data and plot data
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'input_method': 'Upload ZIP File',
        'zarr_root_path': '',
        'dat': None, # For ZIP loaded data (npdict)
        'zarr_store_obj': None, # For Zarr loaded data (root zarr group)
        'level1_groups': [], # e.g., Recordings_AB, Cochlear_recordings
        'selected_level1': None,
        'level2_groups': [], # e.g., Emotionally expressive speech (EmoHI)
        'selected_level2': None,
        'level3_groups': [], # e.g., t1_sad
        'selected_level3': None,
        'recording_ids': [], # Actual recording group, e.g., AB_recording_XXXX
        'selected_recording_id': None,
        'system_type': None,
        'segment_keys': [], # Segments within the selected recording_id
        'selected_segment_key': None,
        'plot_image_bytes': None,
        'audio_file_path': None, # This will store the path to the WAV file
        'download_pulse_times': None,
        'download_pulse_amplitudes': None,
        'download_pulse_parameters': None,
        'temp_dir': None,
        'uploaded_file_name': None
    }

# --- Input Method Selection ---
st.session_state.app_state['input_method'] = st.radio(
    "Select Input Method:",
    ['Upload ZIP File', 'Load from Local Zarr Library'],
    index=0 if st.session_state.app_state['input_method'] == 'Upload ZIP File' else 1,
    key='input_method_radio'
)

# --- Logic for "Upload ZIP File" ---
uploaded_file = None
if st.session_state.app_state['input_method'] == 'Upload ZIP File':
    # If data is not loaded (or if we switched from Zarr to ZIP) display uploader
    if st.session_state.app_state['dat'] is None or st.session_state.app_state['zarr_store_obj'] is not None:
        uploaded_file = st.file_uploader("Upload CI Recording ZIP File", type="zip", key="main_file_uploader")
    else:
        # If data is already loaded from ZIP, offer to upload new
        st.sidebar.markdown("---")
        if st.sidebar.button("Upload New ZIP File", key="clear_upload_button_zip"):
            # Reset relevant state for new ZIP upload
            st.session_state.app_state = {
                'input_method': 'Upload ZIP File', 'zarr_root_path': '',
                'dat': None, 'zarr_store_obj': None, 'level1_groups': [],
                'selected_level1': None, 'level2_groups': [], 'selected_level2': None,
                'level3_groups': [], 'selected_level3': None, 'recording_ids': [],
                'selected_recording_id': None, 'system_type': None, 'segment_keys': [],
                'selected_segment_key': None, 'plot_image_bytes': None,
                'audio_file_path': None, 'download_pulse_times': None,
                'download_pulse_amplitudes': None, 'download_pulse_parameters': None,
                'temp_dir': None, 'uploaded_file_name': None
            }
            st.rerun()

    if uploaded_file is not None:
        # Only process if it's a new file or we just reset
        if st.session_state.app_state['uploaded_file_name'] != uploaded_file.name or st.session_state.app_state['dat'] is None:
            
            # Clean up old temp directory if it exists
            if st.session_state.app_state['temp_dir'] and os.path.exists(st.session_state.app_state['temp_dir']):
                import shutil
                try:
                    shutil.rmtree(st.session_state.app_state['temp_dir'])
                except OSError as e:
                    st.warning(f"Could not remove old temporary directory: {e}. It might be in use.")
                st.session_state.app_state['temp_dir'] = None

            # Create a new temp directory for the current upload
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
                st.session_state.app_state['zarr_store_obj'] = None # Clear Zarr related state
                st.session_state.app_state['level1_groups'] = []
                st.session_state.app_state['selected_level1'] = None
                st.session_state.app_state['level2_groups'] = []
                st.session_state.app_state['selected_level2'] = None
                st.session_state.app_state['level3_groups'] = []
                st.session_state.app_state['selected_level3'] = None
                st.session_state.app_state['recording_ids'] = []
                st.session_state.app_state['selected_recording_id'] = None
                
                st.session_state.app_state['system_type'] = detect_system_type(uploaded_file.name) # Detect system from ZIP name
                
                if 'rec' in st.session_state.app_state['dat'] and st.session_state.app_state['dat']['rec']:
                    st.session_state.app_state['segment_keys'] = list(st.session_state.app_state['dat']['rec'].keys())
                    st.session_state.app_state['selected_segment_key'] = st.session_state.app_state['segment_keys'][0] if st.session_state.app_state['segment_keys'] else None
                else:
                    st.session_state.app_state['segment_keys'] = []
                    st.session_state.app_state['selected_segment_key'] = None
                    st.error("No recording segments found in the 'rec' folder of the ZIP file.")
                
                # Reset plot and download data for fresh upload
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['download_pulse_times'] = None
                st.session_state.app_state['download_pulse_amplitudes'] = None
                st.session_state.app_state['download_pulse_parameters'] = None
                st.session_state.app_state['audio_file_path'] = None # Reset audio path for new ZIP
                
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during file processing or data loading: {e}")
                st.exception(e)
                # Reset all app state on critical error
                st.session_state.app_state = {
                    'input_method': 'Upload ZIP File', 'zarr_root_path': '',
                    'dat': None, 'zarr_store_obj': None, 'level1_groups': [],
                    'selected_level1': None, 'level2_groups': [], 'selected_level2': None,
                    'level3_groups': [], 'selected_level3': None, 'recording_ids': [],
                    'selected_recording_id': None, 'system_type': None, 'segment_keys': [],
                    'selected_segment_key': None, 'plot_image_bytes': None,
                    'audio_file_path': None, 'download_pulse_times': None,
                    'download_pulse_amplitudes': None, 'download_pulse_parameters': None,
                    'temp_dir': None, 'uploaded_file_name': None
                }
                st.stop()
        else:
            uploaded_file.seek(0)
            
# --- Logic for "Load from Local Zarr Library" ---
elif st.session_state.app_state['input_method'] == 'Load from Local Zarr Library':
    zarr_path_input_raw = st.text_input(
        "Enter Path to Local Zarr Library (e.g., C:/path/to/BSCProject_LIBRARY.zarr):", # Updated example path
        value=st.session_state.app_state['zarr_root_path'],
        key='zarr_path_input'
    )
    st.session_state.app_state['zarr_root_path'] = zarr_path_input_raw.strip('"')

    if st.session_state.app_state['zarr_root_path'] and not st.session_state.app_state['zarr_store_obj']:
        if st.button("Load Zarr Library", key='load_zarr_button'):
            try:
                # Clear previous state from ZIP upload
                st.session_state.app_state['dat'] = None
                st.session_state.app_state['temp_dir'] = None
                st.session_state.app_state['uploaded_file_name'] = None

                # Open the root Zarr store
                root_zarr = zarr.open_group(st.session_state.app_state['zarr_root_path'], mode='r')
                st.session_state.app_state['zarr_store_obj'] = root_zarr
                
                # Populate level 1 groups
                st.session_state.app_state['level1_groups'] = [key for key in root_zarr.keys() if isinstance(root_zarr[key], zarr.Group)]
                if st.session_state.app_state['level1_groups']:
                    st.session_state.app_state['selected_level1'] = st.session_state.app_state['level1_groups'][0]
                else:
                    st.warning("No top-level groups found in the specified Zarr library. Check Zarr structure.")
                    st.session_state.app_state['selected_level1'] = None
                
                # Reset all lower-level selections and data on new Zarr load
                st.session_state.app_state['level2_groups'] = []
                st.session_state.app_state['selected_level2'] = None
                st.session_state.app_state['level3_groups'] = []
                st.session_state.app_state['selected_level3'] = None
                st.session_state.app_state['recording_ids'] = []
                st.session_state.app_state['selected_recording_id'] = None
                st.session_state.app_state['segment_keys'] = []
                st.session_state.app_state['selected_segment_key'] = None
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['audio_file_path'] = None # Reset audio path
                st.session_state.app_state['download_pulse_times'] = None
                st.session_state.app_state['download_pulse_amplitudes'] = None
                st.session_state.app_state['download_pulse_parameters'] = None

                st.success(f"Zarr library loaded from {st.session_state.app_state['zarr_root_path']}.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load Zarr library from {st.session_state.app_state['zarr_root_path']}: {e}")
                st.exception(e)
                # Reset Zarr-related state on error
                st.session_state.app_state['zarr_store_obj'] = None
                st.session_state.app_state['level1_groups'] = []
                st.session_state.app_state['selected_level1'] = None
    elif st.session_state.app_state['zarr_store_obj']:
        st.success(f"Zarr library already loaded from {st.session_state.app_state['zarr_root_path']}.")
        # Offer button to unload/reload Zarr if already loaded
        if st.button("Unload/Change Zarr Library", key="unload_zarr_button"):
            # Reset all Zarr related state
            st.session_state.app_state = {k: None if k not in ['input_method'] else st.session_state.app_state[k] for k in st.session_state.app_state}
            st.session_state.app_state['input_method'] = 'Load from Local Zarr Library'
            st.rerun()


# --- Display Content if Data is Loaded (either from ZIP or Zarr) ---
if st.session_state.app_state['dat'] or st.session_state.app_state['zarr_store_obj']:
    st.sidebar.header("Data & Analysis Options")
    if st.session_state.app_state['system_type']:
        st.sidebar.write(f"Detected System Type: **{st.session_state.app_state['system_type'].upper()}**")

    # --- Zarr Hierarchy Navigation in Sidebar (for Zarr input method) ---
    current_zarr_group_at_level = st.session_state.app_state['zarr_store_obj'] # Starts at root

    if st.session_state.app_state['input_method'] == 'Load from Local Zarr Library' and st.session_state.app_state['zarr_store_obj']:
        
        # Level 1 Selection
        if st.session_state.app_state['level1_groups']:
            selected_level1_from_ui = st.sidebar.selectbox(
                "Select Level 1 Group:",
                options=st.session_state.app_state['level1_groups'],
                index=st.session_state.app_state['level1_groups'].index(st.session_state.app_state['selected_level1']) if st.session_state.app_state['selected_level1'] in st.session_state.app_state['level1_groups'] else 0,
                key='level1_select'
            )
            if selected_level1_from_ui != st.session_state.app_state['selected_level1']:
                st.session_state.app_state['selected_level1'] = selected_level1_from_ui
                st.session_state.app_state['selected_level2'] = None # Reset lower levels
                st.session_state.app_state['selected_level3'] = None
                st.session_state.app_state['selected_recording_id'] = None
                st.session_state.app_state['segment_keys'] = []
                st.session_state.app_state['selected_segment_key'] = None
                st.session_state.app_state['plot_image_bytes'] = None
                st.session_state.app_state['audio_file_path'] = None # Reset audio path
                st.rerun()
            current_zarr_group_at_level = current_zarr_group_at_level.get(st.session_state.app_state['selected_level1'])
        
        # Populate and Select Level 2
        if current_zarr_group_at_level and st.session_state.app_state['selected_level1'] and isinstance(current_zarr_group_at_level, zarr.Group): # Ensure it's a zarr group
            st.session_state.app_state['level2_groups'] = [key for key in current_zarr_group_at_level.keys() if isinstance(current_zarr_group_at_level[key], zarr.Group)]
            if st.session_state.app_state['level2_groups']:
                if st.session_state.app_state['selected_level2'] not in st.session_state.app_state['level2_groups']:
                    st.session_state.app_state['selected_level2'] = st.session_state.app_state['level2_groups'][0]
                selected_level2_from_ui = st.sidebar.selectbox(
                    "Select Level 2 Group:",
                    options=st.session_state.app_state['level2_groups'],
                    index=st.session_state.app_state['level2_groups'].index(st.session_state.app_state['selected_level2']) if st.session_state.app_state['selected_level2'] in st.session_state.app_state['level2_groups'] else 0,
                    key='level2_select'
                )
                if selected_level2_from_ui != st.session_state.app_state['selected_level2']:
                    st.session_state.app_state['selected_level2'] = selected_level2_from_ui
                    st.session_state.app_state['selected_level3'] = None # Reset lower levels
                    st.session_state.app_state['selected_recording_id'] = None
                    st.session_state.app_state['segment_keys'] = []
                    st.session_state.app_state['selected_segment_key'] = None
                    st.session_state.app_state['plot_image_bytes'] = None
                    st.session_state.app_state['audio_file_path'] = None # Reset audio path
                    st.rerun()
                current_zarr_group_at_level = current_zarr_group_at_level.get(st.session_state.app_state['selected_level2'])
            else:
                st.session_state.app_state['selected_level2'] = None
                st.session_state.app_state['level3_groups'] = [] # Clear next levels if no groups found
                st.session_state.app_state['recording_ids'] = []

        # Populate and Select Level 3
        if current_zarr_group_at_level and st.session_state.app_state['selected_level2'] and isinstance(current_zarr_group_at_level, zarr.Group): # Ensure it's a zarr group
            st.session_state.app_state['level3_groups'] = [key for key in current_zarr_group_at_level.keys() if isinstance(current_zarr_group_at_level[key], zarr.Group)]
            if st.session_state.app_state['level3_groups']:
                if st.session_state.app_state['selected_level3'] not in st.session_state.app_state['level3_groups']:
                    st.session_state.app_state['selected_level3'] = st.session_state.app_state['level3_groups'][0]
                selected_level3_from_ui = st.sidebar.selectbox(
                    "Select Level 3 Group:",
                    options=st.session_state.app_state['level3_groups'],
                    index=st.session_state.app_state['level3_groups'].index(st.session_state.app_state['selected_level3']) if st.session_state.app_state['selected_level3'] in st.session_state.app_state['level3_groups'] else 0,
                    key='level3_select'
                )
                if selected_level3_from_ui != st.session_state.app_state['selected_level3']:
                    st.session_state.app_state['selected_level3'] = selected_level3_from_ui
                    st.session_state.app_state['selected_recording_id'] = None # Reset lower level
                    st.session_state.app_state['segment_keys'] = []
                    st.session_state.app_state['selected_segment_key'] = None
                    st.session_state.app_state['plot_image_bytes'] = None
                    st.session_state.app_state['audio_file_path'] = None # Reset audio path
                    st.rerun()
                current_zarr_group_at_level = current_zarr_group_at_level.get(st.session_state.app_state['selected_level3'])
            else:
                st.session_state.app_state['selected_level3'] = None
                st.session_state.app_state['recording_ids'] = [] # Clear next levels if no groups found

        # Populate and Select Recording ID (e.g., AB_recording_XXXX)
        if current_zarr_group_at_level and st.session_state.app_state['selected_level3'] and isinstance(current_zarr_group_at_level, zarr.Group): # Ensure it's a zarr group
            # Filter out known metadata/attribute arrays like '__info__' and 'fs_value'
            st.session_state.app_state['recording_ids'] = [key for key in current_zarr_group_at_level.keys() if isinstance(current_zarr_group_at_level[key], zarr.Group) and key not in ['__info__', 'fs_value']]
            
            if st.session_state.app_state['recording_ids']:
                if st.session_state.app_state['selected_recording_id'] not in st.session_state.app_state['recording_ids']:
                    st.session_state.app_state['selected_recording_id'] = st.session_state.app_state['recording_ids'][0]
                selected_recording_id_from_ui = st.sidebar.selectbox(
                    "Select Recording ID:",
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
                    st.session_state.app_state['audio_file_path'] = None # Reset audio path
                    st.session_state.app_state['download_pulse_times'] = None
                    st.session_state.app_state['download_pulse_amplitudes'] = None
                    st.session_state.app_state['download_pulse_parameters'] = None
                    st.rerun()
                
                # Update current_recording_group to be the actual recording group (e.g., AB_recording_XXXX)
                current_recording_group = current_zarr_group_at_level.get(st.session_state.app_state['selected_recording_id'])

                # Populate segment keys for the selected recording ID (direct children, not under 'segments')
                if current_recording_group and isinstance(current_recording_group, zarr.Group):
                    # Segments are now direct children of the recording_group (e.g., AB_recording_XXXX/t1_sad_s1_u06)
                    # Filter out metadata arrays: __info__ and fs_value
                    st.session_state.app_state['segment_keys'] = [
                        key for key in current_recording_group.keys() 
                        if isinstance(current_recording_group[key], zarr.Group) and key not in ['__info__', 'fs_value']
                    ]
                    if st.session_state.app_state['selected_segment_key'] not in st.session_state.app_state['segment_keys']:
                        st.session_state.app_state['selected_segment_key'] = st.session_state.app_state['segment_keys'][0] if st.session_state.app_state['segment_keys'] else None
                else:
                    st.warning(f"No segment groups found directly under recording '{st.session_state.app_state['selected_recording_id']}'. Check Zarr structure.")
                    st.session_state.app_state['segment_keys'] = []
                    st.session_state.app_state['selected_segment_key'] = None
            else:
                st.session_state.app_state['selected_recording_id'] = None
                st.session_state.app_state['segment_keys'] = []
        else: # If level3 is not selected or no groups
            st.session_state.app_state['recording_ids'] = []
            st.session_state.app_state['selected_recording_id'] = None
            st.session_state.app_state['segment_keys'] = []
            st.session_state.app_state['selected_segment_key'] = None


    # --- Metadata Section (now more flexible) ---
    st.sidebar.markdown("---")
    if st.session_state.app_state['input_method'] == 'Upload ZIP File' and st.session_state.app_state['dat'] and st.session_state.app_state['dat'].get('__info__'):
        with st.sidebar.expander("View Metadata (__info__)"):
            st.json(st.session_state.app_state['dat']['__info__'])
    elif st.session_state.app_state['input_method'] == 'Load from Local Zarr Library' and st.session_state.app_state['zarr_store_obj'] and st.session_state.app_state['selected_recording_id']:
        with st.sidebar.expander(f"View Recording Metadata ({st.session_state.app_state['selected_recording_id']})"):
            # Construct the path to the selected recording group
            current_zarr_recording_group_path = ""
            if st.session_state.app_state['selected_level1']:
                current_zarr_recording_group_path += st.session_state.app_state['selected_level1'] + '/'
            if st.session_state.app_state['selected_level2']:
                current_zarr_recording_group_path += st.session_state.app_state['selected_level2'] + '/'
            if st.session_state.app_state['selected_level3']:
                current_zarr_recording_group_path += st.session_state.app_state['selected_level3'] + '/'
            if st.session_state.app_state['selected_recording_id']:
                current_zarr_recording_group_path += st.session_state.app_state['selected_recording_id'] # No trailing slash for direct group access

            current_zarr_recording_group = st.session_state.app_state['zarr_store_obj'].get(current_zarr_recording_group_path)

            if current_zarr_recording_group and '__info__' in current_zarr_recording_group:
                try:
                    info_zarr_array = current_zarr_recording_group['__info__'][:]
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


    # --- Segment Selection (common to both input methods, always present if recording is selected) ---
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
            st.session_state.app_state['audio_file_path'] = None # Reset audio path
            st.rerun() # Rerun to apply segment change and clear cache properly

        # --- Process and Display Selected Segment ---
        if st.session_state.app_state['selected_segment_key']:
            current_segment_name = st.session_state.app_state['selected_segment_key']
            st.subheader(f"Segment: `{current_segment_name}`")

            # Initialize lists for pulse data and fs
            pulse_times = []
            pulse_amplitudes = []
            pulse_prms = []
            fs = None

            if st.session_state.app_state['input_method'] == 'Upload ZIP File' and st.session_state.app_state['dat']:
                segment_data = st.session_state.app_state['dat']['rec'][current_segment_name]
                fs = st.session_state.app_state['dat'].get('__info__', {}).get('fs_scope') or st.session_state.app_state['dat'].get('__info__', {}).get('fs')
                
                # For ZIP, audio path is relative to temp_dir
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


            elif st.session_state.app_state['input_method'] == 'Load from Local Zarr Library' and st.session_state.app_state['zarr_store_obj']:
                # Construct the full path to the recording group
                current_zarr_recording_group_path = ""
                if st.session_state.app_state['selected_level1']:
                    current_zarr_recording_group_path += st.session_state.app_state['selected_level1'] + '/'
                if st.session_state.app_state['selected_level2']:
                    current_zarr_recording_group_path += st.session_state.app_state['selected_level2'] + '/'
                if st.session_state.app_state['selected_level3']:
                    current_zarr_recording_group_path += st.session_state.app_state['selected_level3'] + '/'
                if st.session_state.app_state['selected_recording_id']:
                    current_zarr_recording_group_path += st.session_state.app_state['selected_recording_id'] # No trailing slash for direct group access

                current_zarr_recording_group = st.session_state.app_state['zarr_store_obj'].get(current_zarr_recording_group_path)
                
                if current_zarr_recording_group:
                    # Get fs from recording group
                    if 'fs_value' in current_zarr_recording_group:
                        try:
                            fs = float(current_zarr_recording_group['fs_value'][:].item())
                        except Exception as fs_e:
                            st.warning(f"Could not load 'fs_value' from Zarr for this recording: {fs_e}. Using default 44100.")
                            fs = 44100
                    else:
                        st.warning("Sampling rate 'fs_value' not found in Zarr recording metadata. Using default 44100.")
                        fs = 44100 # Fallback default
                    
                    # The segment group is a direct child of the recording group
                    zarr_segment_group = current_zarr_recording_group.get(current_segment_name)

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
                        for serialized_item in pulse_prms_channels_zarr:
                            if isinstance(serialized_item, str):
                                try:
                                    deserialized_item = json.loads(serialized_item)
                                except (json.JSONDecodeError, ValueError):
                                    deserialized_item = serialized_item
                            else:
                                deserialized_item = serialized_item
                            
                            if isinstance(deserialized_item, list):
                                pulse_prms.append(deserialized_item)
                            else:
                                pulse_prms.append([deserialized_item])
                        
                        # --- Retrieve audio_path from Zarr attributes ---
                        audio_path_from_zarr = zarr_segment_group.attrs.get('audio_path')
                        if audio_path_from_zarr and os.path.exists(audio_path_from_zarr):
                            st.session_state.app_state['audio_file_path'] = audio_path_from_zarr
                        else:
                            st.session_state.app_state['audio_file_path'] = None
                            if audio_path_from_zarr: # Path was stored but doesn't exist
                                st.info(f"Audio path '{audio_path_from_zarr}' for segment '{current_segment_name}' not found on disk.")


                    else:
                        st.error(f"Segment '{current_segment_name}' not found in the selected recording group.")
                        st.stop()
                else:
                    st.error(f"Recording group for '{st.session_state.app_state['selected_recording_id']}' not found in Zarr store.")
                    st.stop()

            else: # Should not happen, but a safeguard
                st.error("Internal Error: No data source selected or available for segment loading.")
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
                        title=f"{current_segment_name} ({detect_system_type(st.session_state.app_state['selected_recording_id']).replace('ab', 'Advanced Bionics').replace('cochlear', 'Cochlear')}): Vectorized Electrodogram", # Use detector for display
                        reverse_channels=(detect_system_type(st.session_state.app_state['selected_recording_id']) == SYSTEM_COCHLEAR) # Use detector for condition
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

                # Store data arrays for download regardless of source (ZIP or Zarr)
                st.session_state.app_state['download_pulse_times'] = pulse_times # As list of arrays
                st.session_state.app_state['download_pulse_amplitudes'] = pulse_amplitudes # As list of arrays
                st.session_state.app_state['download_pulse_parameters'] = pulse_prms # As list of deserialized objects/dicts (list of lists of objects/dicts)
                
                st.image(st.session_state.app_state['plot_image_bytes'])
                st.success("Electrodogram generated successfully!")
            else:
                st.image(st.session_state.app_state['plot_image_bytes'])
                st.info("Electrodogram loaded from cache.")
            
            # Display Download Buttons for Electrodogram and Data Arrays (common to both input methods)
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
                    all_prms_flat = []
                    for channel_prms_list in st.session_state.app_state['download_pulse_parameters']:
                        if isinstance(channel_prms_list, list):
                            all_prms_flat.extend(channel_prms_list)
                        else:
                            all_prms_flat.append(channel_prms_list)

                    if all_prms_flat:
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

