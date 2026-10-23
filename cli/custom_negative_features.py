#!/usr/bin/env python
"""Convert one 16 kHz mono WAV into microWakeWord negative features.

Same layout as voxpopuli_features.py and the stock kahrendt/microwakeword speech
negatives: 10 ms uint16 microfrontend spectrograms, the first 25 frames of each
chunk dropped, then split into non-overlapping 5 s (500 frame) windows. Long
recordings (a whole video or podcast) are read in 60 s chunks so memory stays flat.
"""

import os
import sys
from argparse import ArgumentParser

# Feature extraction is CPU-only; keep TensorFlow (imported by audio_utils) off the GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import soundfile as sf
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.audio_utils import generate_features_for_clip

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 60 * SAMPLE_RATE
MIN_CHUNK_SAMPLES = 6 * SAMPLE_RATE
DROP_FRAMES = 25
WINDOW_FRAMES = 500
# pymicro-features returns spectrograms pre-scaled by this factor; undo it to
# store raw uint16 values exactly like the stock negative datasets.
FEATURE_SCALE = 0.0390625

parser = ArgumentParser()
parser.add_argument("--wav", required=True, help="16 kHz mono WAV")
parser.add_argument("--out-dir", required=True, help="Output RaggedMmap directory")
args = parser.parse_args()


def spectrogram_windows(wav_path: str):
    info = sf.info(wav_path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        sys.exit(f"{wav_path}: expected 16 kHz mono, got {info.samplerate} Hz x {info.channels}")
    chunks = 0
    windows = 0
    for samples in sf.blocks(wav_path, blocksize=CHUNK_SAMPLES, dtype="float32", always_2d=False):
        # A short tail would give no full window; the stock sets repeat short clips the same way.
        if samples.size < MIN_CHUNK_SAMPLES:
            if samples.size == 0:
                continue
            samples = np.tile(samples, -(-MIN_CHUNK_SAMPLES // samples.size))
        features = generate_features_for_clip(samples, step_ms=10, use_c=True)
        features = np.clip(np.rint(features / FEATURE_SCALE), 0, 65535).astype(np.uint16)
        features = features[DROP_FRAMES:]
        chunks += 1
        for start in range(0, features.shape[0] - WINDOW_FRAMES + 1, WINDOW_FRAMES):
            windows += 1
            yield features[start : start + WINDOW_FRAMES]
        if chunks % 30 == 0:
            print(f"      {chunks} min: {windows} windows", flush=True)
    print(f"      {chunks} min total: {windows} windows", flush=True)


os.makedirs(os.path.dirname(os.path.abspath(args.out_dir)), exist_ok=True)
RaggedMmap.from_generator(
    out_dir=args.out_dir,
    sample_generator=spectrogram_windows(args.wav),
    batch_size=1000,
    verbose=False,
)
sys.exit(0)
