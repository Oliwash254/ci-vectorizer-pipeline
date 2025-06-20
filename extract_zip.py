import zipfile
import os

zip_path = r"C:\CI library project\data\Bscproject_library\Recordings_AB\Emotionally expressive speech (EmoHI)\t1_sad\AB_recording_20250522_2057.zip"
extract_to = r"C:\CI library project\data\Bscproject_library\Recordings_AB\Emotionally expressive speech (EmoHI)\t1_sad\AB_recording_20250522_2057"

# Ensure destination exists
os.makedirs(extract_to, exist_ok=True)

# Extract with safe filename handling
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        safe_name = member.replace('"', '').replace(":", '').replace('\\', '/')
        target_path = os.path.join(extract_to, *safe_name.split('/'))

        if not target_path.endswith('/'):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'wb') as f:
                f.write(zip_ref.read(member))

print("✅ Extraction completed safely.")
