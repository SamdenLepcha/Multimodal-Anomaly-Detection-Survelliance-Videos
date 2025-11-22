import os
import uuid
import subprocess
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

PRETRAINED_I3D = os.path.join(
    BASE_DIR, "..", "I3D_Feature_Extraction_resnet", "pretrained", "i3d_r50_kinetics.pth"
)

# Path to your existing I3D extractor script
I3D_EXTRACTOR = os.path.join(
    BASE_DIR, "..", "I3D_Feature_Extraction_resnet", "main.py"
)

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")


# --------------------------------------------------------------
# Save uploaded file
# --------------------------------------------------------------
UPLOAD_DIR = "static/uploads"

def save_uploaded_video(file, upload_dir=UPLOAD_DIR):
    """
    Saves the uploaded video using its ORIGINAL filename.
    If the file exists, it overwrites it.
    """
    os.makedirs(upload_dir, exist_ok=True)

    # Get the exact filename as uploaded
    original_name = file.filename

    out_path = os.path.join(upload_dir, original_name)

    # Save directly
    file.save(out_path)

    return out_path


# --------------------------------------------------------------
# Extract I3D visual features for a single video
# --------------------------------------------------------------
def extract_visual_features(video_path):
    """
    Runs I3D extractor on a single uploaded MP4.
    The extractor expects a directory containing videos, not a single file.
    """

    # ---------------------------
    # 1. Create temporary directory
    # ---------------------------
    temp_dir = "/tmp/i3d_single_video/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Copy uploaded MP4 → temp_dir
    temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
    shutil.copy(video_path, temp_video_path)

    # ---------------------------
    # 2. Ensure output directory
    # ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------
    # 3. Build correct command
    # ---------------------------
    cmd = [
        "python",
        I3D_EXTRACTOR,
        f"--datasetpath={temp_dir}",                # <-- FIXED
        f"--outputpath={OUTPUT_DIR}",
        f"--pretrainedpath={PRETRAINED_I3D}"
    ]

    print("[INFO] Running I3D extractor:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    print("\n---- I3D STDOUT ----")
    print(result.stdout)
    print("---- I3D STDERR ----")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"I3D extractor failed:\n{result.stderr}")

    # ---------------------------
    # 4. Find generated .npy file
    # ---------------------------
    base = os.path.splitext(os.path.basename(video_path))[0]  # "Explosion013"
    npy_pattern = os.path.join(OUTPUT_DIR, f"{base}*.npy")
    matches = glob.glob(npy_pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"[ERROR] No output file produced: {npy_pattern}")

    return matches[0]

