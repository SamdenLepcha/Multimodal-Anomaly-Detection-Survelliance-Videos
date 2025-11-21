from flask import Flask, render_template, request
import os
from models.tevad_runner import TEVADRunner
from utils.video_utils import extract_10crop_i3d, extract_text_emb
from utils.feature_utils import save_plot

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

tevad = TEVADRunner(ckpt_path="ckpt/your_model.pth", device="cuda")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # save uploaded video
    file = request.files["video"]
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    # 1. extract visual features
    vis_feat = extract_10crop_i3d(filepath)  # -> [10, T, 2048]

    # 2. text features
    text_feat = extract_text_emb(filepath)   # -> [10, T, 768]

    # 3. TEVAD inference
    scores = tevad.run(vis_feat, text_feat)

    # 4. save visualization
    plot_path = save_plot(scores, OUTPUT_DIR)

    return render_template("result.html",
                           video_name=file.filename,
                           plot_path=plot_path,
                           anomaly="YES" if scores.max() > 0.5 else "NO",
                           max_score=float(scores.max()))
