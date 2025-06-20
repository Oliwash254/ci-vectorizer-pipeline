import argparse
from . import main_pipeline

def run():
    parser = argparse.ArgumentParser(description="CI Dataset Vectorization Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the CI recordings (zip or folder)")
    parser.add_argument("--output", type=str, required=True, help="Path to output Zarr file")
    parser.add_argument("--include_wavs", action="store_true", help="Include waveform audio in Zarr")
    parser.add_argument("--prefer_unzipped", action="store_true", help="Prefer extracted folders if both zip and folders exist")
    parser.add_argument("--allow_missing_wav", action="store_true", help="Skip missing .wav matches instead of warning")

    args = parser.parse_args()

    main_pipeline.main(
        input_path=args.input_dir,
        output_zarr=args.output,
        include_wavs=args.include_wavs,
        prefer_unzipped=args.prefer_unzipped,
        allow_missing_wav=args.allow_missing_wav
    )

if __name__ == "__main__":
    run()
