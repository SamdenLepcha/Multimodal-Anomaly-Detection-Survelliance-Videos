# utils/plot_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt

def save_anomaly_plot(scores, output_folder, filename_prefix="anomaly"):
    """
    Plot anomaly scores over the entire clip.

    Parameters
    ----------
    scores : array-like
        1D list/array of anomaly scores per 'frame'. In your pipeline this
        should be result['frame_scores'], i.e. snippet scores repeated 16x.
    output_folder : str
        Directory to save the plot into.
    filename_prefix : str
        Prefix for the saved PNG filename.
    """

    os.makedirs(output_folder, exist_ok=True)

    scores = np.asarray(scores, dtype=float)
    num_frames = len(scores)

    # x-axis: frame indices 0..N-1
    frame_idx = np.arange(num_frames)

    # Optional safety log: uncomment if you want to debug
    # print(f"[PLOT] num_frames={num_frames}, min={scores.min():.4f}, max={scores.max():.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(frame_idx, scores, linewidth=1.5)
    plt.xlabel("Frame index")
    plt.ylabel("Anomaly score")
    plt.title("Anomaly Scores Across Entire Clip")
    plt.ylim(0.0, 1.0)            # TEVAD scores are sigmoid'd to [0, 1]
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_filename = f"{filename_prefix}_anomaly.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_path
