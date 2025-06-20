# ci-library-project
# 🎧 CI Vectorizer Pipeline

A robust pipeline for processing cochlear implant (CI) recordings into structured Zarr datasets, with a Streamlit interface for exploration and visualization of electrodograms.

---

## 📦 Project Structure

CI_library_project/
├── src/
│ └── ci_processor/
│ ├── main_pipeline.py # Main processing script
│ ├── ci_vectorization/
│ │ ├── npdict.py # .dat/.zip loading logic
│ │ ├── ab_vectorizer.py # AB-specific pulse vectorizer
│ │ ├── cochlear_vectorizer.py # Cochlear-specific vectorizer
│ │ └── vectorizers.py # Base pulse vectorizer logic
│ ├── waveform_utils.py # Preview & waveform utilities
│ └── plot_utils.py # Electrodogram plotting tools
├── streamlit_app/
│ ├── app.py # Streamlit dashboard
│ ├── utils/
│ │ ├── plotting.py # Streamlit-compatible plotting
│ │ └── zarr_loader.py # Zarr loading helpers
├── data/
│ └── Bscproject_library/ # Source .zip / .wav input files
├── output/
│ └── ci_library.zarr # Output Zarr dataset
├── run_pipeline.py # Entry script to trigger pipeline
├── requirements.txt
├── .gitignore
└── README.md # ← You are here

yaml
Copy
Edit

---

## 🚀 Features

- ✅ Automatic recursive processing of `.zip`, `.wav`, or extracted folders
- ✅ Supports both **Cochlear** and **Advanced Bionics (AB)** CI systems
- ✅ Pulse vectorization with oversampling and quantization
- ✅ Electrodogram previews (Matplotlib/Plotly)
- ✅ Output saved as efficient **Zarr archives**
- ✅ Interactive dashboard via **Streamlit**
- ✅ Audio playback, download buttons, metadata viewing, and pulse summaries

---

## 🛠️ Installation

```
# Clone the repository
git clone https://github.com/Oliwash254/ci-vectorizer-pipeline.git
cd ci-vectorizer-pipeline

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt
# ⚙️ Usage
# 1. Run the processing pipeline manually:

python src/ci_processor/main_pipeline.py \
  --input_path "path/to/your.zip" \
  --output "output/ci_library.zarr"
Supports .zip, _rec_ folders, or .dat directly.

Automatically matches .wav files by similarity.

Saves waveform/electrodogram previews in output/plots/.

# 2. Launch the Streamlit app:
streamlit run streamlit_app/app.py
You can:

Browse recordings

View electrodograms

Download pulse arrays as .npz

Upload new recordings (auto-runs pipeline)

# 📊 Electrodogram Example

# 📁 Data Format
Each segment is processed into Zarr arrays:

pulse_times: 1D float array

pulse_times_channel: matching channel IDs

pulse_amplitudes: amplitude values

pulse_amplitudes_channel: corresponding channels

Metadata includes system, wav_duration, and sample rates

🧪 Development Notes
Vectorizer logic lives in vectorizers.py and system-specific subclasses.

AB and Cochlear differ in polarity, symmetry, and interphase gap handling.

Uses oversampling and interpolation to align pulse shapes.

# 📄 License
This project is part of a bachelor's thesis and is open for academic collaboration. You may use, adapt, or extend the code with attribution.

# 👤 Author
Oliver Shaban — GitHub @Oliwash254

# 🧠 Future Work
Add support for multi-wav per session

Waveform alignment with pulses

Export to EDF or CSV formats

Enhanced filtering and search in UI

✅ Checklist
 Zarr output format

 CI system detection

 Electrodogram plotting

 WAV matching

 Streamlit dashboard

 Upload + rerun support
