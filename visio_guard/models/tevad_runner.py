import torch
import numpy as np

from TEVAD.model import RTFM  # adjust to your actual TEVAD model import
from TEVAD.utils import process_feat


class TEVADRunner:
    def __init__(self):

        # Load trained TEVAD checkpoint
        ckpt_path = "TEVAD/ckpt/latest.pth"   # modify as needed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RTFM()   # matches your TEVAD architecture
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

    def detect(self, visual_feat, text_feat):
        """
        visual_feat: np.ndarray shape [10, T, 2048]
        text_feat:   np.ndarray shape [10, T, 768]
        """

        # Convert raw features → 32-frame segments
        vis_32 = [process_feat(visual_feat[i], 32) for i in range(10)]
        txt_32 = [process_feat(text_feat[i], 32) for i in range(10)]

        vis_32 = torch.tensor(np.array(vis_32), dtype=torch.float32).unsqueeze(0)
        txt_32 = torch.tensor(np.array(txt_32), dtype=torch.float32).unsqueeze(0)

        vis_32 = vis_32.to(self.device)
        txt_32 = txt_32.to(self.device)

        # TEVAD forward pass
        with torch.no_grad():
            _, _, _, _, _, _, logits, _, _, feat_mag = self.model(vis_32, txt_32)

        # feat_mag shape: [10, 32]
        if feat_mag.dim() == 3:
            feat_mag = feat_mag.squeeze(0)

        # Max across 10 crops
        snippet_scores = feat_mag.max(dim=0)[0].cpu().numpy()

        # Decide anomaly threshold (simple)
        anomaly_flag = bool(snippet_scores.max() > 0.5)

        return snippet_scores, anomaly_flag
