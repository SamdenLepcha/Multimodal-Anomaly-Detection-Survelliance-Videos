from flask import Flask, render_template, request, redirect, url_for
import os
import uuid

from models.tevad_runner import TEVADRunner
from utils.video_utils import save_uploaded_video
from utils.feature_utils import extract_i3d_features, extract_text_features
from utils.plot_utils import save_plot


app = Flask(__name__)

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"

tevad = TEVADRunner()   # load TEVAD model once


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No file uploaded", 400

    file = request.files["video"]
    if file.filename == "":
        return "Empty filename", 400

    # Save uploaded video
    video_path = save_uploaded_video(file, UPLOAD_DIR)

    # STEP 1 — Visual I3D features
    visual_feat = extract_i3d_features(video_path)

    # STEP 2 — Caption/Text features (SwinBERT)
    text_feat = extract_text_features(video_path)

    # STEP 3 — TEVAD Anomaly Inference
    scores, anomaly_flag = tevad.detect(visual_feat, text_feat)

    # STEP 4 — Save anomaly curve plot
    plot_path = save_plot(scores, OUTPUT_DIR)

    return render_template(
        "result.html",
        video_path=video_path,
        plot_path=plot_path,
        anomaly=anomaly_flag,
        scores=scores.tolist()
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
