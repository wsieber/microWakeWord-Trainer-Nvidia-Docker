#!/usr/bin/env python3
"""Persistent-process MOSS-TTS-Nano voice-cloning worker."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from moss_tts_nano.defaults import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
)


MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"
# The nano audio tokenizer emits 48000 / 3840 = 12.5 frames per second.
MOSS_FRAMES_PER_SECOND = 12.5
MOSS_MAX_NEW_FRAMES = 64


def max_new_frames(item: dict) -> int:
    """Stop decoding just past the clip's rejection length.

    Reference QA rejects anything longer than maximum_duration, so frames
    beyond it are wasted work; clips that finish earlier are unaffected.
    """

    try:
        maximum_duration = float(item["maximum_duration"])
    except (KeyError, TypeError, ValueError):
        return MOSS_MAX_NEW_FRAMES
    frames = math.ceil(maximum_duration * MOSS_FRAMES_PER_SECOND) + 2
    return max(1, min(MOSS_MAX_NEW_FRAMES, frames))


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def visible_gpus() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [gpu.strip() for gpu in configured.split(",") if gpu.strip()]
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def worker_gpus(entry_count: int) -> list[str]:
    """GPUs to split generation across; MOSS_GPUS caps the count.

    MOSS decodes one clip at a time and a single worker already keeps a GPU
    busy, so extra workers only help when each gets its own GPU.
    """

    gpus = visible_gpus()
    try:
        limit = int(os.environ.get("MOSS_GPUS", len(gpus)))
    except ValueError:
        limit = len(gpus)
    return gpus[: max(1, min(limit, entry_count))]


def run_sharded(args: argparse.Namespace, entries: list[dict], gpus: list[str]) -> int:
    """Run one worker per GPU, each on a round-robin share of the entries."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    shard_paths = []
    for index, gpu in enumerate(gpus):
        shard_path = args.output_dir / f".moss_shard_{index}.jsonl"
        shard_path.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries[index :: len(gpus)]),
            encoding="utf-8",
        )
        shard_paths.append(shard_path)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--input-jsonl",
                    str(shard_path),
                    "--output-dir",
                    str(args.output_dir),
                    "--checkpoint",
                    args.checkpoint,
                    "--audio-tokenizer",
                    args.audio_tokenizer,
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "MOSS_GPUS": "1"},
            )
        )
    print(f"MOSS generating {len(entries)} clip(s) across {len(gpus)} GPUs", flush=True)
    returncodes = [process.wait() for process in processes]
    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)
    return next((code for code in returncodes if code != 0), 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument(
        "--audio-tokenizer",
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
    )
    args = parser.parse_args()

    entries = read_jsonl(args.input_jsonl)
    if not entries:
        return 0

    gpus = worker_gpus(len(entries))
    if len(gpus) > 1:
        return run_sharded(args, entries, gpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa")
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(entries, start=1):
        seed = int(item.get("seed", index))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        output_path = args.output_dir / f"{item['id']}.wav"
        model.inference(
            text=str(item["text"]),
            output_audio_path=str(output_path),
            mode="voice_clone",
            prompt_audio_path=str(item["ref_audio"]),
            reference_audio_path=None,
            text_tokenizer_path=None,
            audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
            audio_tokenizer_pretrained_name_or_path=args.audio_tokenizer,
            device=device,
            nq=None,
            max_new_frames=max_new_frames(item),
            voice_clone_max_text_tokens=32,
            voice_clone_max_memory_per_sample_gb=1.0,
            do_sample=True,
            use_kv_cache=True,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=0.7,
            audio_top_p=0.9,
            audio_top_k=25,
            audio_repetition_penalty=1.3,
        )
        if index % 10 == 0 or index == len(entries):
            print(f"MOSS generated {index}/{len(entries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
