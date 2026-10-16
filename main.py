import os

from configs import default_params as config
from pipeline import run_full_pipeline


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    nd2_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith(".nd2")]
    if not nd2_files:
        print(f"[!] No .nd2 files found in {config.INPUT_DIR}")
        return

    processed, skipped, failed = [], [], []

    for filename in nd2_files:
        input_path = os.path.join(config.INPUT_DIR, filename)

        if os.path.getsize(input_path) < (config.MIN_SIZE_MB * 1024 * 1024):
            skipped.append(filename)
            continue

        print(f"\n[>] Processing: {filename}")
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(config.OUTPUT_DIR, video_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            result_info = run_full_pipeline(input_path, output_dir)
            if result_info is None:
                failed.append(filename)
                continue
            processed.append(filename)
        except Exception as e:
            import traceback
            print(f"[!] Error processing {filename}: {e}")
            traceback.print_exc()
            failed.append(filename)
            continue

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Processed: {len(processed)}")
    print(f"  Skipped (< {config.MIN_SIZE_MB} MB): {len(skipped)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print("  Failed files:")
        for f in failed:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
