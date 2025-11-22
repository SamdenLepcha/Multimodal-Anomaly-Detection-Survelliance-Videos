import subprocess
import os

# ---------------------------------------
# PATHS (update only if folder changes)
# ---------------------------------------

SWINBERT_ROOT = "/home/ubuntu/uca-virginia/Multimodal-Anomaly-Detection-Survelliance-Videos/SwinBERT"
OUTPUT_CAPTION_FILE = f"{SWINBERT_ROOT}/outputs/captions_dense.txt"
OUTPUT_SENT_EMB_DIR = f"{SWINBERT_ROOT}/outputs/sent_emb_n/"
BEST_MODEL = f"{SWINBERT_ROOT}/models/table1/vatex/best-checkpoint/model.bin"
CHECKPOINT = f"{SWINBERT_ROOT}/models/table1/vatex/best-checkpoint/"


# ---------------------------------------
# 1. RUN SWINBERT DENSE CAPTIONING
# ---------------------------------------

def run_swinbert_captioning(video_dir):
    """
    Runs SwinBERT dense captioning inside SwinBERT's environment.
    video_dir = directory containing the uploaded videos
    """

    print("🔵 [SwinBERT] Starting dense caption generation...")

    command = (
        f"cd {SWINBERT_ROOT} && "
        f"python ./src/tasks/dense_caption_mass.py "
        f"--resume_checkpoint {BEST_MODEL} "
        f"--eval_model_dir {CHECKPOINT} "
        f"--dataset_path {video_dir} "
        f"--caption_file {OUTPUT_CAPTION_FILE} "
        f"--file_type video "
        f"--file_format mp4 "
        f"--do_lower_case "
        f"--dense_caption "
        f"--do_test "
    )

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ SwinBERT captioning failed:\n{e}")

    print("🟢 [SwinBERT] Dense captions generated successfully!")


# ---------------------------------------
# 2. READ GENERATED CAPTIONS
# ---------------------------------------

def read_swinbert_captions():
    """
    Reads captions from SwinBERT/outputs/captions_dense.txt
    Returns a list of caption strings.
    """

    if not os.path.exists(OUTPUT_CAPTION_FILE):
        print("⚠️ [SwinBERT] caption file not found!")
        return []

    with open(OUTPUT_CAPTION_FILE, "r") as f:
        captions = [line.strip() for line in f if line.strip()]

    print(f"🟢 [SwinBERT] Loaded {len(captions)} captions.")
    return captions


# ---------------------------------------
# 3. RUN SWINBERT SENTENCE EMBEDDING (generate_caption_se.py)
# ---------------------------------------

def run_swinbert_sentence_embeddings(caption_path, output_dir):
    """
    Runs generate_caption_se.py inside SwinBERT to create SBERT embeddings.
    caption_path = SwinBERT/outputs/captions_dense.txt
    output_dir = SwinBERT/outputs/sent_emb_n/
    """

    print("🔵 [SwinBERT] Starting sentence embedding generation...")

    command = (
        f"cd {SWINBERT_ROOT} && "
        f"python src/tasks/generate_caption_se.py "
        f"--dataset ucf "
        f"--caption_path {caption_path} "
        f"--output_path {output_dir} "
    )

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Sentence embedding generation failed:\n{e}")

    print(f"🟢 [SwinBERT] SBERT embeddings created at: {OUTPUT_SENT_EMB_DIR} ")
