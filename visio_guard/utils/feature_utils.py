import numpy as np

def extract_i3d_features(video_path):
    """
    Placeholder – integrate your actual I3D extractor.
    Expected output: np.ndarray shape [10, T, 2048]
    """
    raise NotImplementedError("Integrate I3D feature extractor here")


def extract_text_features(video_path):
    """
    Placeholder – integrate SwinBERT caption embeddings.
    Expected output: np.ndarray shape [10, T, 768]
    """
    raise NotImplementedError("Integrate SwinBERT extractor here")
