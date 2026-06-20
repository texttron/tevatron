"""Write bf16-sharded copies of fp32 full-finetune checkpoints.

Standalone CLI, run after training as a post-processing step (like
`merge_lora.py` — kept out of the trainer so the training script stays simple).

Why this exists: HF Trainer FSDP `FULL_STATE_DICT` saves write a single,
monolithic, fp32 `model.safetensors` (~30GB for an 8B model). Two problems when
serving that checkpoint across a pool of N backends on one node:
  - fp32 is 2x the bytes bf16 inference needs, and
  - one file means N processes contend on a single inode — load is serialized
    by the filesystem, not parallel.
An 8-way HF pool took ~14 min to come healthy off such a checkpoint, vs ~1s for
a bf16-sharded one (the LoRA-merge path already shards, so it loaded instantly).

This pass casts fp32->bf16 and re-shards into 5GB pieces, so N loaders
parallelize across files. The result is a plain, self-contained model dir that
loads through the normal vLLM / HF path. The source checkpoint is left intact.

Reuses the same sharding/aux-copy/config-patch helpers as merge_lora.py.

Usage:
    # one checkpoint -> sibling -bf16/
    python -m tevatron.utils.bf16_save /path/to/checkpoint-10137

    # explicit destination
    python -m tevatron.utils.bf16_save /path/to/checkpoint-10137 --dst /path/to/out

    # every checkpoint-*/ under a run dir -> checkpoint-*-bf16/
    python -m tevatron.utils.bf16_save /path/to/output/qwen3-8b-reranker-hf-fsdp
"""
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open

from tevatron.utils.merge_lora import _shard_and_write


def convert_to_bf16(src_dir: Path | str, dst_dir: Path | str) -> int:
    """Write a bf16-sharded copy of the checkpoint at src_dir into dst_dir.

    Streams tensors from src_dir's .safetensors, casts fp32->bf16, and writes
    new 5GB-sharded files into dst_dir (+ index). Non-weight files (config,
    tokenizer, chat_template, etc.) are copied; training-state files are
    skipped. The source checkpoint is left untouched. Idempotent — a dst that
    already has model.safetensors.index.json is left as-is.

    Returns the number of weight tensors written.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if (dst_dir / "model.safetensors.index.json").exists():
        return 0  # already converted

    src_files = sorted(src_dir.glob("*.safetensors"))
    if not src_files:
        raise FileNotFoundError(f"No safetensors files in {src_dir}")

    n = 0

    def _iter():
        nonlocal n
        for sf_path in src_files:
            with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    n += 1
                    yield key, f.get_tensor(key)  # _shard_and_write casts fp32->bf16

    _shard_and_write(_iter(), dst_dir, src_dir)
    return n


def _is_checkpoint_dir(p: Path) -> bool:
    """A model checkpoint dir (has weights) that isn't already a -bf16 output."""
    return (
        p.is_dir()
        and not p.name.endswith("-bf16")
        and any(p.glob("*.safetensors"))
    )


def convert_all_checkpoints(run_dir: Path | str) -> list[str]:
    """Convert every checkpoint-*/ under run_dir to a sibling checkpoint-*-bf16/.

    Returns the list of source checkpoint dirs processed. Idempotent per-ckpt.
    """
    run_dir = Path(run_dir)
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint-*") if _is_checkpoint_dir(p)),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    done: list[str] = []
    for ckpt in ckpts:
        convert_to_bf16(ckpt, f"{ckpt}-bf16")
        done.append(str(ckpt))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write bf16-sharded copies of fp32 checkpoints (fast N-way serving load)."
    )
    ap.add_argument("path", help="A single checkpoint dir, OR a run dir "
                                 "(converts every checkpoint-*/ inside).")
    ap.add_argument("--dst", default=None,
                    help="Destination for single-checkpoint mode. Omit to default to "
                         "a sibling <ckpt>-bf16/ (single ckpt) or per-checkpoint -bf16 (run dir).")
    args = ap.parse_args()

    path = Path(args.path)
    if args.dst:
        n = convert_to_bf16(path, args.dst)
        print(f"Wrote {n} tensors (bf16-sharded) -> {args.dst}")
    elif _is_checkpoint_dir(path):
        dst = str(path).rstrip("/") + "-bf16"
        n = convert_to_bf16(path, dst)
        print(f"Wrote {n} tensors (bf16-sharded) -> {dst}")
    else:
        done = convert_all_checkpoints(path)
        if not done:
            print(f"No checkpoint-*/ dirs with weights found under {path}")
        else:
            print(f"Converted {len(done)} checkpoint(s):")
            for d in done:
                print(f"  {d} -> {d}-bf16")


if __name__ == "__main__":
    main()
