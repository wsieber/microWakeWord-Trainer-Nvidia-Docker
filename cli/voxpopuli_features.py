#!/usr/bin/env python
"""Convert one VoxPopuli parquet shard into microWakeWord negative features.

Matches the stock kahrendt/microwakeword speech negatives: 10 ms uint16
microfrontend spectrograms, clips repeated to at least 6 s, the first 25
frames dropped, then split into non-overlapping 5 s (500 frame) windows.
"""

import io
import os
import sys
from argparse import ArgumentParser

# Feature extraction is CPU-only; keep TensorFlow (imported by audio_utils) off the GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.audio_utils import generate_features_for_clip

SAMPLE_RATE = 16000
MIN_CLIP_SAMPLES = 6 * SAMPLE_RATE
DROP_FRAMES = 25
WINDOW_FRAMES = 500
# pymicro-features returns spectrograms pre-scaled by this factor; undo it to
# store raw uint16 values exactly like the stock negative datasets.
FEATURE_SCALE = 0.0390625

parser = ArgumentParser()
parser.add_argument("--parquet", required=True, help="VoxPopuli parquet shard")
parser.add_argument("--out-dir", required=True, help="Output RaggedMmap directory")
args = parser.parse_args()


def spectrogram_windows(parquet_path: str):
    shard = pq.ParquetFile(parquet_path)
    clips = 0
    windows = 0
    for group in range(shard.num_row_groups):
        audio_column = shard.read_row_group(group, columns=["audio"]).column("audio")
        for entry in audio_column:
            audio = entry.as_py()
            if not audio or not audio.get("bytes"):
                continue
            # VoxPopuli WAVs are 32-bit float; read as float so libsndfile doesn't
            # truncate them to int16 (generate_features_for_clip scales floats itself).
            samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE or samples.size == 0:
                continue
            if samples.ndim > 1:
                samples = samples[:, 0]
            if samples.size < MIN_CLIP_SAMPLES:
                samples = np.tile(samples, -(-MIN_CLIP_SAMPLES // samples.size))

            features = generate_features_for_clip(samples, step_ms=10, use_c=True)
            features = np.clip(np.rint(features / FEATURE_SCALE), 0, 65535).astype(np.uint16)
            features = features[DROP_FRAMES:]
            clips += 1
            for start in range(0, features.shape[0] - WINDOW_FRAMES + 1, WINDOW_FRAMES):
                windows += 1
                yield features[start : start + WINDOW_FRAMES]
        print(f"      row group {group + 1}/{shard.num_row_groups}: {clips} clips, {windows} windows", flush=True)


os.makedirs(os.path.dirname(os.path.abspath(args.out_dir)), exist_ok=True)
RaggedMmap.from_generator(
    out_dir=args.out_dir,
    sample_generator=spectrogram_windows(args.parquet),
    batch_size=1000,
    verbose=False,
)
sys.exit(0)
