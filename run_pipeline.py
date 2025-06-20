# run_pipeline.py

import subprocess
from pathlib import Path
import sys

def run():
    project_root = Path(__file__).parent
    input_dir = project_root / "data" / "Bscproject_library"
    output_path = project_root / "output" / "ci_library.zarr"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,  # This ensures it uses the current Python/venv
        "-m", "src.ci_processor.main_pipeline",
        "--input_dir", str(input_dir),
        "--output", str(output_path)
    ]

    print(f"🚀 Running pipeline with:")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_path}")
    print("🔧 Command:", " ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Pipeline completed successfully.")
    else:
        print("❌ Pipeline failed.")
        print("STDERR:\n", result.stderr)
        print("STDOUT:\n", result.stdout)

if __name__ == "__main__":
    run()
