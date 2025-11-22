import os
import numpy as np
import torch
from models.model import Model
import argparse

class SimpleArgs:
    def __init__(self):
        self.feature_size = 2048       
        self.emb_dim = 768             
        self.fusion = 'concat'         
        self.feature_group = 'both'    
        self.aggregate_text = True     
        self.batch_size = 1
        self.dataset = "ucf"           

def load_features(i3d_path, emb_path):
    # 1. LOAD RAW DATA
    visual = np.load(i3d_path).astype(np.float32)
    text = np.load(emb_path).astype(np.float32)

    print("[DEBUG] raw visual:", visual.shape)
    print("[DEBUG] raw text:  ", text.shape)

    # =========================================================
    # STEP 1: FIX VISUAL ORIENTATION: want (B, 10, T, 2048)
    # =========================================================
    # Case: extractor gives (T, 10, 2048)
    if visual.ndim == 3 and visual.shape[1] == 10 and visual.shape[0] != 10:
        print(f"[Fix] Swapping time and crop dims for visual: {visual.shape} -> ", end="")
        # (T, 10, 2048) -> (10, T, 2048)
        visual = visual.transpose(1, 0, 2)
        print(visual.shape)

    # Ensure last dim is 2048 (just in case)
    shape = visual.shape
    if 2048 in shape:
        axis_2048 = list(shape).index(2048)
        if axis_2048 != visual.ndim - 1:
            print(f"[Fix] Transposing visual to put 2048 last: {shape} -> ", end="")
            if visual.ndim == 2:
                visual = visual.T  # (2048, T) -> (T, 2048)
            elif visual.ndim == 3:
                perm = list(range(visual.ndim))
                perm.pop(axis_2048)
                perm.append(axis_2048)
                visual = visual.transpose(*perm)
            print(visual.shape)

    # =========================================================
    # STEP 2: FORCE VISUAL TO (1, 10, T, 2048)
    # =========================================================
    if visual.ndim == 2:  # (T, 2048)
        visual = visual[None, None, :, :]        # (1, 1, T, 2048)
        visual = np.tile(visual, (1, 10, 1, 1))  # (1, 10, T, 2048)

    elif visual.ndim == 3:
        if visual.shape[0] == 10:
            # (10, T, 2048) -> (1, 10, T, 2048)
            visual = visual[None, :, :, :]
        elif visual.shape[0] == 1:
            # (1, T, 2048) -> (1, 1, T, 2048) -> tile to 10 crops
            visual = visual[:, None, :, :]
            visual = np.tile(visual, (1, 10, 1, 1))
        else:
            # fallback: treat dim0 as crop dim
            visual = visual[None, :, :, :]       # (1, C, T, 2048)

    elif visual.ndim == 4:
        # assume already (B, C, T, 2048)-like
        if visual.shape[0] != 1:
            visual = visual[:1]
        if visual.shape[1] == 1:
            visual = np.tile(visual, (1, 10, 1, 1))

    if visual.ndim != 4:
        raise ValueError(f"Visual features shape mismatch. Needed 4 dims, got {visual.shape}")

    # Now expect visual: (1, 10, T, 2048)
    target_T = visual.shape[2]
    print("[DEBUG] visual after reshape:", visual.shape)

    # =========================================================
    # STEP 3: FIX TEXT ORIENTATION: want (T, 768)
    # =========================================================
    text = np.squeeze(text)

    if text.ndim == 1:
        if text.shape[0] == 768:
            text = text[None, :]  # (1, 768)
        else:
            raise ValueError(f"Unexpected 1D text shape: {text.shape}")

    elif text.ndim == 2:
        if text.shape[1] == 768:
            # (T, 768) already
            pass
        elif text.shape[0] == 768:
            print(f"[Fix] Transposing text from {text.shape} to (T, 768)")
            text = text.T
        else:
            raise ValueError(f"Unexpected 2D text shape: {text.shape}")
    else:
        raise ValueError(f"Unexpected text ndim: {text.ndim}, shape={text.shape}")

    current_T = text.shape[0]

    # =========================================================
    # STEP 4: ALIGN TEXT LENGTH TO VISUAL T
    # =========================================================
    if current_T != target_T:
        print(f"[Note] Aligning Text ({current_T}) to Visual ({target_T})")
        indices = np.linspace(0, current_T - 1, target_T).astype(int)
        text = text[indices]

    # (T, 768) -> (1, 10, T, 768)
    text = text[None, None, :, :]
    text = np.tile(text, (1, 10, 1, 1))

    print("[DEBUG] final visual:", visual.shape)  # expect (1, 10, T, 2048)
    print("[DEBUG] final text:  ", text.shape)    # expect (1, 10, T, 768)

    return visual, text


def run_single_inference(video_name, model_path, feature_dir, emb_dir, threshold=0.5):
    basename = os.path.splitext(video_name)[0]
    i3d_path = os.path.join(feature_dir, basename + ".npy")
    emb_path = os.path.join(emb_dir, basename + "_emb.npy")

    if not os.path.exists(i3d_path):
        raise FileNotFoundError(f"Visual features missing: {i3d_path}")
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Text embeddings missing: {emb_path}")

    print(f"[✓] Loading features for: {basename}")
    print(">>> i3d_path:", i3d_path)
    print(">>> emb_path:", emb_path)
    print(">>> visual raw shape:", np.load(i3d_path).shape)
    print(">>> text  raw shape:", np.load(emb_path).shape)

    visual, text = load_features(i3d_path, emb_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    visual = torch.tensor(visual).float().to(device)
    text = torch.tensor(text).float().to(device)

    args = SimpleArgs()
    model = Model(args).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        ret_values = model(visual, text)
        logits = ret_values[6]

        if logits.dim() == 3:
            logits = logits.mean(dim=1)  # (B, 10, T, 1) -> (B, T, 1)

        logits = logits.view(-1)  # (T,)
        snippet_scores = logits.cpu().numpy().tolist()

    if not snippet_scores:
        snippet_scores = [0.0] * visual.shape[2]

    frame_scores = np.repeat(snippet_scores, 16).tolist()
    max_s = max(snippet_scores) if len(snippet_scores) > 0 else 0.0

    return {
        "basename": basename,
        "snippet_scores": snippet_scores,
        "frame_scores": frame_scores,
        "verdict": "ANOMALY" if max_s > threshold else "NORMAL",
        "max_score": max_s
    }

    
# Main block remains the same...
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--model", type=str, default="best-model.pkl")
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = run_single_inference(args.video_name, args.model, args.feature_dir, args.emb_dir, args.threshold)
    print(f"Verdict: {result['verdict']}")