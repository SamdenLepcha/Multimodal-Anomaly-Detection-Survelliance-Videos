import matplotlib.pyplot as plt
import numpy as np
import os

def save_plot(scores, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "anomaly_curve.png")

    plt.figure(figsize=(10,4))
    plt.plot(scores, label="Anomaly score")
    plt.ylim([0, 1])
    plt.xlabel("Snippet index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path
