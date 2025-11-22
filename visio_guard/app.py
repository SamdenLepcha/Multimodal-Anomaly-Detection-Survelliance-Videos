import os
import numpy as np
from flask import Flask, render_template, request

# --- UTILS ---
from utils.video_utils import save_uploaded_video, extract_visual_features
from utils.captioner import run_swinbert_captioning, read_swinbert_captions, run_swinbert_sentence_embeddings
from utils.file_utils import move_latest_embedding_file
from utils.plot_utils import save_anomaly_plot

# --- MODELS ---
from models.tevad_runner import load_features
from models.tevad_runner import run_single_inference

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
OUTPUT_FOLDER = "static/outputs/"
MODEL_PATH = 'models/best-model.pkl'

app_path = '/home/ubuntu/uca-virginia/Multimodal-Anomaly-Detection-Survelliance-Videos/visio_guard/'
caption_path = "/home/ubuntu/uca-virginia/Multimodal-Anomaly-Detection-Survelliance-Videos/SwinBERT/outputs/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(caption_path+"sent_emb_n/", exist_ok=True)


# ========================================================
# HOME PAGE
# ========================================================
@app.route("/")
def index():
    return render_template("index.html")


# ========================================================
# UPLOAD + TEVAD PIPELINE
# ========================================================
@app.route("/upload", methods=["POST"])
def upload():

    # =======================================================
    # 1. VALIDATE UPLOADED FILE
    # =======================================================
    
    if "video" not in request.files:
        return "No video uploaded."
    file = request.files["video"]
    if file.filename == "":
        return "No file selected."

    # =======================================================
    # 2. SAVE VIDEO
    # =======================================================
    
    saved_video_path = save_uploaded_video(file, UPLOAD_FOLDER)
    print(f"[✓] Uploaded video saved at: {saved_video_path}")

    # =======================================================
    # 3. EXTRACT I3D FEATURES
    # =======================================================
    
    try:
        i3d_path = extract_visual_features(saved_video_path)
        print(f"[✓] I3D features saved at: {i3d_path}")
    except Exception as e:
        return f"I3D feature extraction failed:<br>{str(e)}"

    # ============================================================
    # 3. RUN CAPTION GENERATION (SwinBERT)
    # ============================================================
    try:
        print("\n=== [✓] Running SwinBERT Dense Captioning for data: ===", app_path+saved_video_path)
        run_swinbert_captioning(app_path+UPLOAD_FOLDER)
        
    except Exception as e:
        return f"SwinBERT Caption Generation failed:<br>{str(e)}"
    try:
        print("\n=== [✓] Reading Captions ===")
        captions = read_swinbert_captions()
        print("First few captions:", captions[:3])
    
        print(f"\n=== [✓] Generating Sentence Embeddings===")
        run_swinbert_sentence_embeddings(caption_path + "captions_dense.txt", caption_path+"sent_emb_n/")
            
    except Exception as e:
        return f"Embedding Generation Failed:<br>{str(e)}"
    try:
        # =======================================================
        # 4. Move embedding into Flask outputs using file_utils
        # =======================================================
    
        caption_emb_path = move_latest_embedding_file(caption_path+"sent_emb_n/", OUTPUT_FOLDER)
        print(f"[✓] Final embedding copied to: {caption_emb_path}")
    
    except Exception as e:
        print("❌ Embedding file not present:", str(e))

    # =======================================================
    # 5. TEVAD INFERENCE (Corrected Block)
    # =======================================================
    
    scores = np.zeros(100) 
    verdict = "ERROR"
    
    try:
        print(f"[✓] Running TEVAD Inference on {file.filename}...")
        
        result = run_single_inference(
            video_name=file.filename, 
            model_path=MODEL_PATH, 
            feature_dir=OUTPUT_FOLDER, 
            emb_dir=OUTPUT_FOLDER, 
            threshold=0.6
        )

        # If successful, overwrite the defaults
        scores = np.array(result['frame_scores'])
        verdict = result['verdict']
        print("[✓] TEVAD inference complete.")

    except Exception as e:
        print("❌ TEVAD FAILED. USING FALLBACK.")
        print("Error:", str(e))

    # =======================================================
    # 6. SAVE PLOT AND RENDER
    # =======================================================

    print("[DEBUG] len(frame_scores):", len(result['frame_scores']))
    print("[DEBUG] first 10 frame_scores:", result['frame_scores'][:10])

    basename = os.path.splitext(file.filename)[0]
    plot_path = save_anomaly_plot(scores, OUTPUT_FOLDER, filename_prefix=basename)
    plot_filename = os.path.basename(plot_path)

    return render_template(
            "report.html",
            video_filename=os.path.basename(saved_video_path),
            scores=scores.tolist(),
            plot_filename=plot_filename,
            verdict=verdict
    )


# ========================================================
# RUN SERVER
# ========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
