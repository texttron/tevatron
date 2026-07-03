"""Merge a LoRA adapter into its base model and write a bf16-sharded checkpoint.

Standalone CLI, run after training. The merged output is a plain, fully
self-contained model directory — it loads through the normal vLLM / HF path
(just point at the dir), with NO `--enable-lora` / `--lora-modules` plumbing.

Why this exists: serving a raw PEFT adapter via vLLM's runtime LoRA-merge path
is fragile — a merged-LoRA reranker that scored 0.82 NDCG@10 under HF DDP scored
0.02 under vLLM's runtime merge (see docs/bugs-and-fixes.md). Pre-merging here
makes the checkpoint a vanilla model and removes that failure mode entirely.

Tensor-level merge (no model class instantiated): architecture-agnostic and
low-memory (one base tensor at a time). For each base weight W a LoRA targets:
    W_merged = W + (alpha / r) * (B @ A)      [rslora: alpha / sqrt(r)]

Only needed for the HF-PEFT LoRA path. The Megatron (megatron-bridge) LoRA path
already saves fully-merged weights at checkpoint time, so it needs no merge.

Usage:
    # one adapter checkpoint -> sibling -merged-bf16/
    python -m tevatron.utils.merge_lora /path/to/checkpoint-2000

    # explicit destination
    python -m tevatron.utils.merge_lora /path/to/checkpoint-2000 --dst /path/to/out

    # every adapter checkpoint-*/ under a run dir -> checkpoint-*-merged-bf16/
    python -m tevatron.utils.merge_lora /path/to/output/qwen3-8b-reranker-hf-ddp-lora

    # override base (adapter trained against a moved/renamed base)
    python -m tevatron.utils.merge_lora /path/to/ckpt --base /fsx/.../Qwen3-8B-Base
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB shards

# Files that must never be copied into a merged checkpoint (weights handled
# separately; training state is irrelevant to inference).
_SKIP_AUX = {
    "model.safetensors.index.json", "adapter_model.safetensors",
    "adapter_model.bin", "adapter_config.json", "optimizer.bin",
    "pytorch_model_fsdp.bin", "scheduler.pt", "trainer_state.json",
}
_SKIP_AUX_PREFIX = ("rng_state",)


def _shard_and_write(tensors_iter, dst_dir: Path, aux_src: Path) -> None:
    """Stream (name, tensor) pairs into 5GB bf16 shards + index, then copy
    config/tokenizer aux files from aux_src. fp32 tensors are cast to bf16."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    out_idx = 1
    template = "model-{idx:05d}-of-XXXXX.safetensors"
    out_names: list[str] = []

    def flush():
        nonlocal cur, cur_bytes, out_idx
        if not cur:
            return
        fname = template.format(idx=out_idx)
        save_file(cur, str(dst_dir / fname), metadata={"format": "pt"})
        for name in cur:
            weight_map[name] = fname
        out_names.append(fname)
        cur = {}
        cur_bytes = 0
        out_idx += 1

    for key, t in tensors_iter:
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)
        cur[key] = t
        nbytes = t.numel() * t.element_size()
        cur_bytes += nbytes
        total_bytes += nbytes
        if cur_bytes >= SHARD_BYTES:
            flush()
    flush()

    n = len(out_names)
    rename = {old: old.replace("XXXXX", f"{n:05d}") for old in out_names}
    for old, new in rename.items():
        (dst_dir / old).rename(dst_dir / new)
    weight_map = {k: rename[v] for k, v in weight_map.items()}

    with (dst_dir / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)

    _copy_aux(aux_src, dst_dir)
    _patch_config_dtype(dst_dir)


def _copy_aux(src_dir: Path, dst_dir: Path) -> None:
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix == ".safetensors" or f.name in _SKIP_AUX:
            continue
        if f.name.startswith(_SKIP_AUX_PREFIX):
            continue
        dst = dst_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def _patch_config_dtype(dst_dir: Path) -> None:
    cfg_path = dst_dir / "config.json"
    if not cfg_path.exists():
        return
    with cfg_path.open() as f:
        cfg = json.load(f)
    if "dtype" in cfg:
        cfg["dtype"] = "bfloat16"
    cfg["torch_dtype"] = "bfloat16"
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)


def merge_lora_to_bf16(
    adapter_dir: Path | str,
    dst_dir: Path | str,
    base_dir: Path | str | None = None,
) -> int:
    """Merge a PEFT LoRA adapter into its base model, write a bf16-sharded dir.

    Returns the number of merged modules. Idempotent (skips a dst that already
    has an index). base_dir defaults to the adapter_config's
    base_model_name_or_path.
    """
    adapter_dir = Path(adapter_dir)
    dst_dir = Path(dst_dir)
    if (dst_dir / "model.safetensors.index.json").exists():
        return 0

    with (adapter_dir / "adapter_config.json").open() as f:
        acfg = json.load(f)
    if acfg.get("peft_type") != "LORA":
        raise ValueError(f"Unsupported peft_type {acfg.get('peft_type')!r} (only LORA)")
    if base_dir is None:
        base_dir = acfg["base_model_name_or_path"]
    base_dir = Path(base_dir)

    r = acfg["r"]
    alpha = acfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if acfg.get("use_rslora", False) else alpha / r

    # Load adapter A/B tensors (small), keyed by the base weight they modify.
    # PEFT key: base_model.model.<base_key_without_.weight>.lora_{A,B}.weight
    # Also collect `modules_to_save` weights (e.g. a SEQ_CLS `score` head), which
    # PEFT stores verbatim under base_model.model.<name>.weight (NOT as A/B). These
    # are *trained* full weights that must REPLACE / be ADDED to the base; missing
    # them yields a randomly-initialized head -> garbage scores (the same failure
    # class as loading a seq-cls ckpt as causal-LM). They live in the adapter as
    # either "<name>.modules_to_save.default.weight" or plain "<name>.weight".
    lora_A: dict[str, torch.Tensor] = {}
    lora_B: dict[str, torch.Tensor] = {}
    saved_modules: dict[str, torch.Tensor] = {}  # base_key -> trained full weight
    adp = adapter_dir / "adapter_model.safetensors"
    if not adp.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
    with safe_open(str(adp), framework="pt", device="cpu") as f:
        for k in f.keys():
            body = k[len("base_model.model."):]
            if body.endswith(".lora_A.weight"):
                lora_A[body[: -len(".lora_A.weight")] + ".weight"] = f.get_tensor(k)
            elif body.endswith(".lora_B.weight"):
                lora_B[body[: -len(".lora_B.weight")] + ".weight"] = f.get_tensor(k)
            elif ".lora_" not in body and "lora_embedding" not in body:
                # A modules_to_save full weight. Strip PEFT's ".modules_to_save.
                # default" infix if present so the key matches the base/model key.
                base_key = body.replace(".modules_to_save.default.", ".")
                saved_modules[base_key] = f.get_tensor(k).to(torch.bfloat16)
    targets = set(lora_A) & set(lora_B)
    if not targets:
        raise ValueError(f"No LoRA A/B pairs found in {adapter_dir}")

    base_files = sorted(base_dir.glob("*.safetensors"))
    if not base_files:
        raise FileNotFoundError(f"No safetensors files in base {base_dir}")

    merged = 0
    saved_emitted: set[str] = set()

    def _iter():
        nonlocal merged
        for sf in base_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for key in f.keys():
                    # A trained modules_to_save weight overrides the base tensor
                    # outright (e.g. SEQ_CLS `score` replaces base `lm_head`/score).
                    if key in saved_modules:
                        saved_emitted.add(key)
                        yield key, saved_modules[key]
                        continue
                    t = f.get_tensor(key)
                    if key in targets:
                        A = lora_A[key].to(torch.float32)
                        B = lora_B[key].to(torch.float32)
                        t = (t.to(torch.float32) + (B @ A) * scale).to(torch.bfloat16)
                        merged += 1
                    yield key, t
        # Emit any trained module the base didn't already have under that key
        # (e.g. `score.weight` when the base is a plain CausalLM with `lm_head`).
        for key, t in saved_modules.items():
            if key not in saved_emitted:
                yield key, t

    # Config/tokenizer come from base; overlay adapter-dir aux (tokenizer the
    # training callback saved) without clobbering base config.
    _shard_and_write(_iter(), dst_dir, base_dir)
    for f in adapter_dir.iterdir():
        if not f.is_file() or f.suffix == ".safetensors":
            continue
        if f.name in _SKIP_AUX or f.name.startswith(_SKIP_AUX_PREFIX) or f.name == "config.json":
            continue
        dst = dst_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    if merged != len(targets):
        raise RuntimeError(
            f"Merged {merged} modules but adapter has {len(targets)} targets — "
            f"key mapping mismatch (base/adapter architecture differ?)."
        )
    return merged


def _is_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists()


def merge_all_checkpoints(run_dir: Path | str, base: str | None = None) -> list[str]:
    """Merge every adapter checkpoint-*/ under run_dir -> -merged-bf16 sibling."""
    run_dir = Path(run_dir)
    ckpts = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    ckpts = [p for p in ckpts if p.is_dir() and _is_adapter_dir(p)
             and not p.name.endswith(("-bf16", "-merged-bf16"))]
    for ck in ckpts:
        merge_lora_to_bf16(ck, f"{ck}-merged-bf16", base_dir=base)
    return [str(p) for p in ckpts]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge LoRA adapter(s) into base, save bf16-sharded (vLLM-loadable)."
    )
    ap.add_argument("path", help="A single adapter checkpoint dir, OR a run dir "
                                 "(merges every adapter checkpoint-*/ inside).")
    ap.add_argument("--dst", default=None,
                    help="Destination for single-adapter mode. Omit to default to "
                         "a sibling <ckpt>-merged-bf16/.")
    ap.add_argument("--base", default=None,
                    help="Base model dir. Defaults to adapter_config's base_model_name_or_path.")
    args = ap.parse_args()

    path = Path(args.path)
    if args.dst:
        n = merge_lora_to_bf16(path, args.dst, base_dir=args.base)
        print(f"Merged {n} modules -> {args.dst}")
    elif _is_adapter_dir(path):
        dst = str(path).rstrip("/") + "-merged-bf16"
        n = merge_lora_to_bf16(path, dst, base_dir=args.base)
        print(f"Merged {n} modules -> {dst}")
    else:
        done = merge_all_checkpoints(path, base=args.base)
        if not done:
            print(f"No adapter checkpoint-*/ dirs found under {path}")
        else:
            print(f"Merged {len(done)} adapter checkpoint(s):")
            for d in done:
                print(f"  {d} -> {d}-merged-bf16")


if __name__ == "__main__":
    main()
