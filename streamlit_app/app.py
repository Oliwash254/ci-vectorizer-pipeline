import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
import tempfile
import traceback
import io # Import io for byte stream handling
import sys
from pathlib import Path

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
    from ci_processor.electrodogram import plot_electrodogram # Import the function directly
    from ci_processor.ci_vectorization.vectorizers import SYSTEM_COCHLEAR # Import for reverse_channels logic

except ImportError as e:
    st.error(f"Failed to import ci_processor modules. Please ensure your 'ci_processor' "
             f"package is correctly installed and accessible in your Python environment. "
             f"Error: {e}")
    st.stop()


# --- Function to detect system type (copied from main_pipeline.py) ---
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
        # Fallback or a more robust detection could be added here if needed
        st.warning("Could not reliably detect system type from file name. Defaulting to 'ab'. "
                   "Please ensure 'cochlear' or 'ab' is in your ZIP file name for auto-detection.")
        return "ab" # Default to 'ab' if not found


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="CI Electrodogram Visualizer")

st.title("🦻 CI Electrodogram Visualizer")
st.markdown("""
Upload your CI recording `.zip` file (containing `rec` and `__info__` folders)
to visualize the vectorized electrodogram.
""")

uploaded_file = st.file_uploader("Upload CI Recording ZIP File", type="zip")

# Initialize session state for storing plot data
if 'plot_data_cache' not in st.session_state:
    st.session_state.plot_data_cache = {}

if uploaded_file is not None:
    # Use a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, uploaded_file.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        try:
            # 1. Load data from ZIP
            # Add a unique key for the uploaded file to clear cache if a new file is uploaded
            uploaded_file_hash = hash(uploaded_file.read()) # Simple hash for cache invalidation
            if 'last_uploaded_file_hash' not in st.session_state or st.session_state.last_uploaded_file_hash != uploaded_file_hash:
                st.session_state.plot_data_cache = {} # Clear cache if new file
                st.session_state.last_uploaded_file_hash = uploaded_file_hash
            
            uploaded_file.seek(0) # Rewind the buffer after hashing

            with st.spinner("Loading data... This may take a moment."):
                dat = zip_to_npdict(zip_path, prefer_unzipped=True) # Prefer unzipped in case of issues

            if not dat or 'rec' not in dat or not dat['rec']:
                st.error("Could not load recording data from the ZIP file. "
                         "Please ensure it contains a 'rec' folder with channel data.")
                st.stop()

            st.sidebar.header("Processing Options")

            # 2. Detect System Type
            system_type = detect_system_type(uploaded_file.name) # Use uploaded_file.name for path-based detection
            st.sidebar.write(f"Detected System Type: **{system_type.upper()}**")

            # Initialize vectorizer based on detected system type
            vectorizer_instance = get_vectorizer(system_type)

            # 3. Segment Selection
            segment_keys = list(dat['rec'].keys())
            if not segment_keys:
                st.error("No recording segments found in the 'rec' folder of the ZIP file.")
                st.stop()

            selected_segment_key = st.sidebar.selectbox(
                "Select a Recording Segment:",
                options=segment_keys,
                help="Choose which segment's electrodogram to visualize.",
                key="segment_select" # Add a key to the selectbox for better state management
            )

            # 4. Process selected segment and manage cache
            if selected_segment_key:
                st.subheader(f"Electrodogram for Segment: `{selected_segment_key}`")
                
                # Check if plot data is already in cache for this segment
                if selected_segment_key in st.session_state.plot_data_cache:
                    st.write("Displaying cached electrodogram...")
                    plot_buffer = st.session_state.plot_data_cache[selected_segment_key]
                    st.image(plot_buffer) # Display the image from buffer
                else:
                    # If not in cache, proceed with full processing
                    st.write("Generating new electrodogram...")
                    segment_data = dat['rec'][selected_segment_key]
                    if not isinstance(segment_data, dict) or not all(isinstance(v, np.ndarray) for v in segment_data.values()):
                        st.error(f"Segment data for '{selected_segment_key}' is not in the expected format (dictionary of numpy arrays).")
                        st.stop()
                    
                    # Sort channel keys numerically if they are digits, otherwise alphabetically
                    sorted_channel_keys = sorted(segment_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
                    X_list = [segment_data[ch] for ch in sorted_channel_keys]
                    
                    # Pad all channel arrays to the maximum length to create a uniform 2D array
                    max_len = max(len(arr) for arr in X_list)
                    X_padded = [np.pad(arr, (0, max_len - len(arr))) for arr in X_list]
                    X = np.array(X_padded) # This will be (n_channels, n_samples)

                    fs = dat.get('__info__', {}).get('fs_scope') # Try 'fs_scope' first
                    if fs is None: # Fallback to 'fs' if 'fs_scope' not found
                        fs = dat.get('__info__', {}).get('fs')
                    
                    if fs is None:
                        st.error("Sampling rate ('fs_scope' or 'fs') could not be determined. Cannot proceed.")
                        st.stop()

                    with st.spinner(f"Vectorizing and plotting electrodogram for '{selected_segment_key}'..."):
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
                                title=f"{selected_segment_key} ({system_type.replace('ab', 'Advanced Bionics').replace('cochlear', 'Cochlear')}): Vectorized Electrodogram",
                                reverse_channels=(system_type == "cochlear")
                            )
                        else:
                            ax.set_title(f"Vectorized Electrodogram: {selected_segment_key} (No Pulses Found)")
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Channel')

                        # Save the figure to a BytesIO object and store in cache
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        st.session_state.plot_data_cache[selected_segment_key] = buf.getvalue() # Store bytes in cache
                        
                        st.pyplot(fig) # Display the plot directly
                        
                    plt.close(fig) # Close the figure to free memory

                # --- Download Button (always displayed after plot generation/retrieval) ---
                if selected_segment_key in st.session_state.plot_data_cache:
                    plot_bytes = st.session_state.plot_data_cache[selected_segment_key]
                    st.download_button(
                        label="Download Electrodogram as PNG",
                        data=plot_bytes,
                        file_name=f"{selected_segment_key}_electrodogram.png",
                        mime="image/png",
                        key=f"download_button_{selected_segment_key}" # Unique key for each download button
                    )
                    st.success("Electrodogram displayed and ready for download!")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e) # Display full traceback in Streamlit
            st.write("Please check the format of your ZIP file and its contents.")
