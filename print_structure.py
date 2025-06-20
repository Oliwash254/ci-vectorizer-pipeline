import zipfile
from pathlib import Path

def inspect_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print(f"📦 Contents of: {zip_path}")
        for name in zf.namelist():
            print("  -", name)

# Run it
inspect_zip("data/Bscproject_library/AB_recording_20250522_2057.zip")
