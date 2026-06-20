# Tevatron Megatron Installation Guide

This guide documents installing tevatron with Megatron support for MoE reranker training on a SLURM cluster (8x H200 per node, CUDA 12.8+, shared `/fsx` filesystem).

## Prerequisites

- NVIDIA H200/H100 GPUs (SM 9.0)
- CUDA driver >= 12.8
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Build from a **compute node** (GPU access required for CUDA source builds)

## 1. Create Isolated venv

```bash
cd /path/to/tevatron
uv venv .venv --python 3.12
source .venv/bin/activate
```

## 2. Install PyTorch

Install build tools from PyPI first, then torch from the PyTorch index:

```bash
uv pip install packaging ninja setuptools wheel cmake pybind11

uv pip install "torch>=2.10.0,<3.0" --index-url https://download.pytorch.org/whl/cu128
```

**Use `--index-url` (singular), not `--extra-index-url` for torch.** Otherwise pip may pick a `+cu130` wheel that's incompatible with the system `nvcc` (12.8). Build tools must be installed separately since they're not hosted on the PyTorch index.

## 3. Install CUDA Source-Build Packages

```bash
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install \
    "flash-attn==2.8.1" \
    --no-build-isolation --no-cache
```

- `TORCH_CUDA_ARCH_LIST=9.0` builds only for H200/H100, cutting compile time significantly.
- flash-attn must be `<= 2.8.1` for TransformerEngine v2.6 compatibility.

## 4. Install nvidia-cuDNN

TransformerEngine needs cuDNN at build time:

```bash
uv pip install nvidia-cudnn-cu12==9.10.2.21
export CUDNN_PATH=$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")
```

Verify: `ls $CUDNN_PATH/lib/libcudnn.so*` should show `libcudnn.so.9`.

## 5. Install TransformerEngine

```bash
NVTE_CUDA_ARCHS="90" NVTE_FRAMEWORK=pytorch MAX_JOBS=32 \
    uv pip install --no-build-isolation --no-deps \
    "git+https://github.com/NVIDIA/TransformerEngine.git@v2.6"
```

**Note:** TransformerEngine uses `NVTE_CUDA_ARCHS` (no dot, `"90"`), NOT `TORCH_CUDA_ARCH_LIST`.

## 6. Install Megatron-Core

For Qwen3.5 models (GDN + MTP + MoE support):
```bash
uv pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0"
```

For Qwen2.5 models (standard attention):
```bash
uv pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1"
```

## 7. Install mbridge (HF ↔ Megatron weight converter)

```bash
uv pip install --no-deps "git+https://github.com/ISEEKYAN/mbridge.git"
```

This is the core bridge package that converts HuggingFace checkpoints to Megatron's parallel layout (TP/PP/EP sharding) on-the-fly. It supports:
- Qwen3.5 (dense + MoE variants)
- Qwen3 (MoE)
- Qwen2/2.5
- LLaMA, Mistral, DeepSeek, Gemma

## 8. Pin Triton

```bash
uv pip install triton==3.3.1
```

TransformerEngine v2.6's MoE token permutation kernels are incompatible with Triton >= 3.6.

## 9. Install Runtime Dependencies

```bash
uv pip install "transformers>=4.45.0" datasets tokenizers peft \
    "numpy<2.0.0" pandas wandb tqdm onnxscript
```

`onnxscript` is required by TransformerEngine's ONNX export module (imported at TE load time).

## 10. Re-pin cuDNN

Some packages in step 9 may override cuDNN. Always re-pin last:

```bash
uv pip install nvidia-cudnn-cu12==9.10.2.21
```

## 11. Install Tevatron

```bash
cd /path/to/tevatron
uv pip install -e . --no-deps
```

## 12. Verify

```bash
python3 -c "
import torch
import megatron.core
import transformer_engine
import mbridge
from tevatron.megatron import MegatronRerankerEngine, MegatronRerankerConfig
print(f'PyTorch: {torch.__version__}')
print(f'Megatron-Core: {megatron.core.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('All good')
"
```

Expected warnings (harmless):
- `Supported flash-attn versions are >= 2.1.1, <= 2.8.1` — TE prints this but works fine
- `Apex is not installed. Falling back to Torch Norm` — TE provides fused norms

## Summary of CUDA Architecture Env Vars

| Package | Env Var | H200 Value | Format |
|---------|---------|------------|--------|
| PyTorch, flash-attn | `TORCH_CUDA_ARCH_LIST` | `9.0` | With dot |
| TransformerEngine | `NVTE_CUDA_ARCHS` | `"90"` | No dot |

## Installation Order

```
1.  torch              (foundation)
2.  flash-attn         (needs torch for CUDA build)
3.  nvidia-cudnn       (TE needs it at build time)
4.  TransformerEngine  (compiles against cudnn)
5.  Megatron-LM        (pure Python, depends on TE at runtime)
6.  mbridge            (pure Python, depends on Megatron at runtime)
7.  triton==3.3.1      (TE MoE kernels break on triton>=3.6)
8.  Other deps         (may override cudnn — re-pin after)
9.  cudnn re-pin       (last step, always)
10. tevatron           (editable, no-deps)
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ninja: no such file or directory` during TE build | `uv pip install ninja` first |
| `Could not find cudnn_LIBRARY` | Install `nvidia-cudnn-cu12` and export `CUDNN_PATH` |
| Stale CMake cache from failed TE build | `rm -rf ~/.cache/uv/git-v0/checkouts/*/build` |
| `RuntimeError: Unsupported function referenced` in MoE forward | Pin `triton==3.3.1` |
| `ImportError: cannot import name 'parse_hybrid_pattern'` | Use `megatron-bridge==0.3.1`, not 0.4.1 |
| OOM during CUDA compilation | Reduce `MAX_JOBS` (try 16 or 8) |

## Quick Launch Test

After installation, verify with a dry run (replace model path):

```bash
torchrun --nproc_per_node=8 -m tevatron.megatron.driver.train \
    --model_name_or_path /path/to/Qwen3.5-35B-A3B \
    --expert_model_parallel_size 8 \
    --train_group_size 4 \
    --micro_batch_size 1 \
    --max_seq_len 128 \
    --total_steps 2 \
    --dataset_path /path/to/small_test.jsonl \
    --save_dir /tmp/test_output
```
