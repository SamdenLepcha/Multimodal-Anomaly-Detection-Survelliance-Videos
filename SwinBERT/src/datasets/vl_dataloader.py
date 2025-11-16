"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
Modified for raw video fine-tuning by Samden Lepcha.
"""

import os
import os.path as op
import torch
from torch.utils.data import Dataset
from src.utils.comm import get_world_size
from .caption_tensorizer import build_tensorizer
from .data_sampler import DistributedSamplerLimited, NodeSplitSampler
from src.utils.logger import LOGGER as logger

# ✅ Import your existing inference utilities
from src.tasks.run_caption_VidSwinBert_inference import _online_video_decode, _transforms
import yaml

def collate_skip_none(batch):
    """Custom collate_fn that safely filters out None samples before collation."""
    # Filter out None samples
    batch = [b for b in batch if b is not None]

    # If all samples are None, return an empty batch
    if len(batch) == 0:
        print("[WARN] Empty batch after filtering bad samples — skipping this batch.")
        return ([], [], [])

    try:
        collated = torch.utils.data.dataloader.default_collate(batch)
    except TypeError:
        # Fallback: manually filter elements that are None inside tuples
        clean_batch = []
        for b in batch:
            if b is None:
                continue
            if isinstance(b, tuple):
                clean_batch.append(tuple(x for x in b if x is not None))
            else:
                clean_batch.append(b)
        collated = torch.utils.data.dataloader.default_collate(clean_batch)  # ✅ FIXED LINE

    # 🩹 Force consistent 3-tuple output (keys, tensors, meta)
    if isinstance(collated, (list, tuple)):
        if len(collated) == 2:
            collated = (collated[0], collated[1], [None] * len(collated[0]))

    return collated

# =====================================================
#   NEW: RawVideoDataset (uses same pipeline as inference)
# =====================================================
class RawVideoDataset(Dataset):
    def __init__(self, args, yaml_file, tokenizer, tensorizer, is_train=True, on_memory=False):
        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.is_train = is_train

        # Load YAML file
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        self.videos = data.get("train_videos", [])
        self.captions = data.get("captions", [])

        assert len(self.videos) == len(self.captions), (
            f"YAML mismatch: {len(self.videos)} videos but {len(self.captions)} captions"
        )

        logger.info(f"Loaded {len(self.videos)} video-caption pairs from {yaml_file}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        caption = self.captions[idx]
        if isinstance(caption, list):
            caption = caption[0]  # pick first if multiple available

        # Decode and preprocess frames (same as inference)
        if not os.path.exists(video_path):
            print(f"[ERROR] Missing video file: {video_path}")
            return None
        
        frames = _online_video_decode(self.args, video_path)
        frames = _transforms(self.args, frames)
        
        if frames is None:
            print(f"[WARN] Failed to decode or transform: {video_path}")
            return None


        # Tensorize the sample for training
        tensorized = self.tensorizer.tensorize_example_e2e(caption, frames)
        return (video_path, tensorized, None)  # consistent with training loop expectations


# =====================================================
#   Build Dataset and DataLoader
# =====================================================
def build_dataset(args, yaml_file, tokenizer, is_train=True):
    logger.info(f'YAML file: {yaml_file}')
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file), f"{yaml_file} does not exist"
    tensorizer = build_tensorizer(args, tokenizer, is_train=is_train)
    dataset_class = RawVideoDataset
    return dataset_class(args, yaml_file, tokenizer, tensorizer, is_train, args.on_memory)


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle, distributed, random_seed, limited_samples=-1):
    if distributed:
        if hasattr(dataset, "is_composite") and dataset.is_composite:
            logger.info("Enable NodeSplitSampler with first_epoch_skip_shuffle=True")
            return NodeSplitSampler(
                dataset, shuffle=shuffle, random_seed=random_seed,
                first_epoch_skip_shuffle=True)
        elif limited_samples < 1:
            return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=random_seed)
        else:
            return DistributedSamplerLimited(dataset, shuffle=shuffle, limited=limited_samples)
    return torch.utils.data.sampler.RandomSampler(dataset) if shuffle else torch.utils.data.sampler.SequentialSampler(dataset)


def make_data_loader(args, yaml_file, tokenizer, is_distributed=True,
                     is_train=True, start_iter=0, num_gpus=8):

    dataset = build_dataset(args, yaml_file, tokenizer, is_train=is_train)

    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info(f"Train with {images_per_gpu} samples per GPU")
        logger.info(f"Total batch size {images_per_batch}")
        logger.info(f"Total training steps {num_iters}")
    else:
        shuffle = False
        images_per_gpu = getattr(args, "per_gpu_eval_batch_size", 1)
        num_iters = None
        start_iter = 0

    limited_samples = getattr(args, "limited_samples", -1) // num_gpus if hasattr(args, "limited_samples") else -1
    random_seed = args.seed

    sampler = make_data_sampler(dataset, shuffle, is_distributed, random_seed, limited_samples)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)

    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True, worker_init_fn=init_seeds,collate_fn=collate_skip_none
    )
    return data_loader


def init_seeds(seed=88):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
