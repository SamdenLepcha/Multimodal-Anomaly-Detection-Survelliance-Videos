import torch
import numpy as np
from model import Model
from utils import process_feat

class TEVADRunner:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = torch.device(device)
        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def run(self, vis_feat_10crop, text_feat_10crop):
        """
        Inputs:
            vis_feat_10crop: np.array [10, T, 2048]
            text_feat_10crop: np.array [10, T, 768]
        """

        vis_32 = []
        txt_32 = []

        for i in range(10):
            vis_32.append(process_feat(vis_feat_10crop[i], 32))
            txt_32.append(process_feat(text_feat_10crop[i], 32))

        vis_32 = torch.tensor(np.array(vis_32)).unsqueeze(0).float().to(self.device)
        txt_32 = torch.tensor(np.array(txt_32)).unsqueeze(0).float().to(self.device)

        # model forward
        with torch.no_grad():
            _, _, _, _, _, _, _, _, _, feat_mag = self.model(vis_32, txt_32)

        feat_mag = feat_mag.squeeze(0)      # [10, 32]
        snippet_scores = feat_mag.max(dim=0)[0]  # [32]

        # upsample to frame-level (16 frames per snippet)
        frame_scores = np.repeat(snippet_scores.cpu().numpy(), 16)

        return frame_scores
