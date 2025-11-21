import matplotlib.pyplot as plt
import torch, sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    auc, roc_curve, precision_recall_curve, average_precision_score
)

from utils import get_gt, process_feat


def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        all_snippet_scores = []

        # -------------------------------------------------------------
        # tqdm PROGRESS BAR — shows how many test videos processed
        # -------------------------------------------------------------
        for i, (video_input, video_text) in enumerate(
            tqdm(dataloader, desc="Testing", ncols=100)
        ):
            # -------------------------------------------------------------
            # Dataloader output shapes (original TEVAD):
            #   video_input: [B, T, 10, 2048]
            #   video_text:  [B, T, 10, 768]
            # We need [B, 10, T, C]
            # -------------------------------------------------------------
            video_input = video_input.to(device)
            video_text = video_text.to(device)

            # Rearrange to [B, 10, T, C]
            video_input = video_input.permute(0, 2, 1, 3)   # [B, 10, T, 2048]
            video_text = video_text.permute(0, 2, 1, 3)    # [B, 10, T, 768]

            # Now shapes are:
            #   video_input: [1, 10, T, 2048]
            #   video_text:  [1, 10, T, 768]
            B, C10, T, F = video_input.shape  # C10 should be 10

            # Move to numpy for process_feat
            video_input_np = video_input.squeeze(0).cpu().numpy()  # [10, T, 2048]
            video_text_np = video_text.squeeze(0).cpu().numpy()    # [10, T, 768]

            # -------------------------------------------------------------
            # FIX 1 — Downsample VIDEO features to 32 segments
            #        Each crop: [T, 2048] → [32, 2048]
            # -------------------------------------------------------------
            video_crops_32 = []
            for crop_idx in range(C10):
                v = process_feat(video_input_np[crop_idx], 32)    # [32, 2048]
                video_crops_32.append(v)
            video_crops_32 = np.stack(video_crops_32, axis=0)     # [10, 32, 2048]

            # -------------------------------------------------------------
            # FIX 2 — Downsample TEXT features to 32 segments
            #        Each crop: [T, 768] → [32, 768]
            # -------------------------------------------------------------
            text_crops_32 = []
            for crop_idx in range(C10):
                t = process_feat(video_text_np[crop_idx], 32)     # [32, 768]
                text_crops_32.append(t)
            text_crops_32 = np.stack(text_crops_32, axis=0)       # [10, 32, 768]

            # -------------------------------------------------------------
            # Convert back to torch: [1, 10, 32, C]
            # -------------------------------------------------------------
            v = torch.from_numpy(video_crops_32).unsqueeze(0).to(device)  # [1,10,32,2048]
            t = torch.from_numpy(text_crops_32).unsqueeze(0).to(device)   # [1,10,32,768]

            # -------------------------------------------------------------
            # Forward pass through model — TSA-safe now (T=32)
            # -------------------------------------------------------------
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, \
            feat_abn_bottom, feat_norm_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(v, t)

            # feat_magnitudes: [1,10,32] or [10,32]
            if feat_magnitudes.dim() == 3:
                feat_magnitudes = feat_magnitudes.squeeze(0)  # [10,32]

            # -------------------------------------------------------------
            # Aggregate 10 crops → per-snippet score (max across crops)
            # -------------------------------------------------------------
            snippet_scores = feat_magnitudes.max(dim=0)[0]        # [32]
            all_snippet_scores.append(snippet_scores.cpu().numpy())

        # =============================================================
        # After all videos processed → compute AUC / AP
        # =============================================================
        all_snippet_scores = np.concatenate(all_snippet_scores, axis=0)

        # Repeat each snippet score 16× (one per frame)
        pred = np.repeat(all_snippet_scores, 16)

        # Ground truth labels (frame-level)
        gt = np.array(get_gt(args.dataset, args.gt))

        # ---------------- Metrics ----------------
        fpr, tpr, _ = roc_curve(gt, pred)
        precision, recall, _ = precision_recall_curve(gt, pred)

        pr_auc = auc(recall, precision)
        rec_auc = auc(fpr, tpr)
        ap = average_precision_score(gt, pred)

        print("ap :", ap)
        print("auc:", rec_auc)

        # ---------------- Visdom Logging ----------------
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)

        # ---------------- Optional Saving ----------------
        if args.save_test_results:
            np.save(f'results/{args.dataset}_pred.npy', pred)
            np.save(f'results/{args.dataset}_fpr.npy', fpr)
            np.save(f'results/{args.dataset}_tpr.npy', tpr)
            np.save(f'results/{args.dataset}_precision.npy', precision)
            np.save(f'results/{args.dataset}_recall.npy', recall)
            np.save(f'results/{args.dataset}_auc.npy', rec_auc)
            np.save(f'results/{args.dataset}_ap.npy', ap)

        return rec_auc, ap
