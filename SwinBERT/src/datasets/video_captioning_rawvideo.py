import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from src.datasets.caption_tensorizer import build_tensorizer
from src.utils.logger import LOGGER as logger
from src.tasks.run_caption_VidSwinBert_inference import _online_video_decode, _transforms

class RawVideoDataset(Dataset):
    def __init__(self, args, yaml_file, tokenizer, tensorizer, is_train=True, on_memory=False):
        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.is_train = is_train
        self.yaml_file = yaml_file

        # Parse YAML manually (since we may not have ruamel)
        import yaml
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        self.videos = data.get("train_videos", [])
        self.captions = data.get("captions", [])
        assert len(self.videos) == len(self.captions), "Mismatch between videos and captions"

        logger.info(f"Loaded {len(self.videos)} video-caption pairs from {yaml_file}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        captions = self.captions[idx]
        frames = _online_video_decode(self.args, video_path)
        frames = _transforms(self.args, frames)
        input_tuple = self.tensorizer.tensorize_example_e2e(captions[0], frames)
        return input_tuple
