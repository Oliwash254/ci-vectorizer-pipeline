import zipfile
from pathlib import Path
import os
import shutil

DATA_DIR = Path("data/Bscproject_library")

def has_required_structure(zip_path):
    """
    Enhanced version that strips extra quotes and normalizes names.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            folders = set()
            for file in zipf.namelist():
                parts = file.strip("/").split("/")
                if len(parts) >= 1:
                    folder = parts[0].strip().lower().replace("\\", "/")
                    folder = folder.strip('"').strip("'")  # strip extra quotes
                    folders.add(folder)

            print(f"\n[🧪] Contents of {zip_path}:")
            for f in sorted(folders):
                print(f"  - {f!r}")

            print(f"[🔍] Normalized folders: {folders}")
            print(f"[🔍] Check: 'rec' in normalized = {'rec' in folders}, '__info__' in normalized = {'__info__' in folders}")

            return "rec" in folders and "__info__" in folders

    except Exception as e:
        print(f"[❌] Error reading {zip_path}: {e}")
        return False

def fix_zip(zip_path):
    stem = zip_path.stem
    extract_path = zip_path.parent / stem

    if not has_required_structure(zip_path):
        print(f"[X] Skipped (missing folders): {zip_path.name}")
        return False

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_path)

    print(f"[~] Extracted: {zip_path.name}")
    return True


def fix_all():
    zip_paths = list(DATA_DIR.glob("*.zip"))
    print(f"[🔍] Found {len(zip_paths)} ZIP file(s) in {DATA_DIR}")

    fixed_count = 0
    for zip_path in zip_paths:
        if fix_zip(zip_path):
            fixed_count += 1

    print(f"\n[✅] Done. Fixed {fixed_count} out of {len(zip_paths)} ZIP file(s).")


if __name__ == "__main__":
    fix_all()
