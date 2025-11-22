import os
import shutil
from glob import glob

def move_latest_embedding_file(src_dir, dst_dir):
    """
    Finds the newest .npy embedding file inside SwinBERT outputs/sent_emb_n/
    and moves it to visio_guard/static/outputs/.
    
    Returns the final destination path.
    """

    # Ensure destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Find all .npy files in the source directory
    files = glob(os.path.join(src_dir, "*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found in: {src_dir}")

    # Pick the most recent .npy file
    latest_file = max(files, key=os.path.getmtime)

    filename = os.path.basename(latest_file)
    dst_path = os.path.join(dst_dir, filename)

    # Move the file
    shutil.move(latest_file, dst_path)

    print(f"🟢 Moved embedding file:\n  {latest_file}\n→ {dst_path}")

    return dst_path
