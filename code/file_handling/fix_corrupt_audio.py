import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# -----------------------------
# CONFIG
# -------------------------
# ----

SOURCE_DIR = Path("../../../../../../mnt/BSP_NAS2_vol3/Video/Video2019/Tomsledge")
OLD_DIR = SOURCE_DIR
FIXED_DIR = Path("../../../../../../mnt/BSP_NAS2_vol3/Video/Video2019/TLOVER")

MOVE_ORIGINALS = False   # True = move AVI files to OLD_DIR
DRY_RUN = False          # True = only print actions
UNPARSEABLE_REPORT = Path("unparseable_datetime_filenames_TLOVER_2019.txt")
UNPARSEABLE_DATE_FOLDER = "unparseable_datetime"
FORCE_REPROCESS = False  # True = ignore existing outputs and process everything again
VERIFY_EXISTING_OUTPUT = False  # True = use ffprobe to validate existing MP4 before skipping

# pattern like: Rost3_20200521_111401.avi
DATE_PATTERN = re.compile(r".*_(\d{8})_(\d{6})\.avi$", re.IGNORECASE)


# -----------------------------
# FUNCTIONS
# -----------------------------

def extract_date(filename):
    m = DATE_PATTERN.match(filename)
    if not m:
        return None

    d = m.group(1)
    t = m.group(2)
    try:
        parsed = datetime.strptime(f"{d}_{t}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None

    return parsed.strftime("%Y-%m-%d")


def fix_video(src, dst):

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags", "+genpts",
        "-i", str(src),
        "-c:v", "copy",
        "-an",
        str(dst)
    ]

    print("Running:", " ".join(cmd))

    if not DRY_RUN:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"ffmpeg failed for {src} (exit code: {result.returncode})")


def is_valid_existing_output(output_file):
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False

    if not VERIFY_EXISTING_OUTPUT:
        return True

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(output_file),
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


# -----------------------------
# MAIN
# -----------------------------

def main():

    avi_files = [p for p in SOURCE_DIR.rglob("*") if p.suffix.lower() == ".avi"]
    unparseable_files = []
    processed_count = 0
    skipped_existing_count = 0
    failed_count = 0
    moved_count = 0

    print(f"Found {len(avi_files)} AVI files")

    for avi in avi_files:

        date = extract_date(avi.name)

        if not date:
            print("Unparseable datetime in filename, using fallback output folder:", avi)
            unparseable_files.append(str(avi))
            date = UNPARSEABLE_DATE_FOLDER

        out_dir = FIXED_DIR / date
        out_dir.mkdir(parents=True, exist_ok=True)

        output_file = out_dir / (avi.stem + ".mp4")

        print(f"\nProcessing: {avi}")
        print(f"Output: {output_file}")

        already_done = (not FORCE_REPROCESS) and is_valid_existing_output(output_file)
        if already_done:
            print("Skipping conversion (already processed):", output_file)
            skipped_existing_count += 1
        else:
            fix_video(avi, output_file)

            if not DRY_RUN and not is_valid_existing_output(output_file):
                print("Conversion did not produce a valid output, leaving source in place:", avi)
                failed_count += 1
                continue

            if not DRY_RUN:
                processed_count += 1

        if MOVE_ORIGINALS:

            old_path = OLD_DIR / avi.relative_to(SOURCE_DIR)
            old_path.parent.mkdir(parents=True, exist_ok=True)

            if old_path.resolve() == avi.resolve():
                print("Skipping move (source and destination are identical):", avi)
                continue

            print("Moving original ->", old_path)

            if not DRY_RUN:
                shutil.move(avi, old_path)
                moved_count += 1

    print("\nSummary")
    print(" - New conversions:", processed_count)
    print(" - Skipped existing outputs:", skipped_existing_count)
    print(" - Failed conversions:", failed_count)
    if MOVE_ORIGINALS:
        print(" - Originals moved:", moved_count)

    print(f"\nFiles with unparseable datetime in filename: {len(unparseable_files)}")
    if unparseable_files:
        for name in unparseable_files:
            print(" -", name)

        if not DRY_RUN:
            UNPARSEABLE_REPORT.parent.mkdir(parents=True, exist_ok=True)
            with UNPARSEABLE_REPORT.open("w", encoding="utf-8") as f:
                f.write("# Files with unparseable datetime in filename\n")
                for name in unparseable_files:
                    f.write(f"{name}\n")
            print(f"Saved unparseable list to: {UNPARSEABLE_REPORT}")
        else:
            print("DRY_RUN enabled: not writing unparseable filename report")


if __name__ == "__main__":
    main()