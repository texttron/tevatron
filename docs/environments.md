# Environments & Dependencies

Tevatron 3.0 spans several workloads with **mutually incompatible dependency
pins** (transformer-engine versions, vLLM's torch ABI, faiss needing conda).
Rather than one fragile mega-environment, we use **separate, single-purpose
environments**, each reproducing one task. This doc says which environment does
what, why it's separate, and how to build it.

> Cluster note: on this machine always `export NVTE_FUSED_ATTN=0` (dual CUDA
> runtime conflict) and keep **numpy pinned at 1.26.4** everywhere (numpy 2.x
> breaks the torch ABI; `pyseismic-lsr`/faiss installs love to bump it).

## At a glance

| Environment | Manager | Used for | Why separate |
|---|---|---|---|
| `.venv` | uv | HF/FSDP1 reranker training; Megatron full-FT **and LoRA**; SPLADE encode; seismic search | the main training env |
| `.venv-eval` | uv | vLLM reranking; teacher annotation; BEIR scoring | vLLM pins a different torch (2.10) than training (2.11) |
| `conda_envs/retriever` | **conda** | dense-retriever eval: **faiss-gpu**, pyserini/BM25, BEIR | faiss-gpu is broken under pip/uv — needs conda |
| `.venv-laconic` | uv | cross-checking the legacy LACONIC stack (transformers 4.51.3) | reproduction-only; not needed for the migrated code |

The version skew between training (`.venv`), vLLM eval (`.venv-eval`), and faiss
(conda) is intentional and load-bearing — do **not** try to merge those three.

### Pinned dependency files

Every environment has a frozen, version-pinned manifest at the repo root, so a
rebuild is reproducible rather than resolver-dependent:

| File | Environment | §  |
|---|---|---|
| `requirements-megatron.txt` | `.venv` — training, TE **2.6** (frozen for paper-run reproduction) | §1, §3 |
| `requirements-lora.txt` | Megatron env, TE **2.12** + `megatron-bridge` (full-FT **and** LoRA — preferred for fresh installs) | §3 |
| `requirements-eval.txt` | `.venv-eval` — vLLM + BEIR + annotation | §4 |
| `requirements-laconic.txt` | `.venv-laconic` — legacy 4.51.3 cross-check (optional) | §5 |
| `environment-retriever.yml` | conda `retriever` — faiss-gpu + pyserini + BEIR | §2 |

Each file's header carries its exact rebuild command. They are point-in-time
freezes (the date is in the header); the git-pinned CUDA source builds
(transformer-engine, megatron-core, mbridge) are pinned by commit/tag inside the
`-megatron`/`-lora` files.

> **A note on `.venv-lora`.** Earlier setups used a separate `.venv-lora` for
> Megatron LoRA, because LoRA needs `megatron-bridge`, which pins
> `transformer-engine>=2.10,<2.13` while `.venv` was *frozen at TE 2.6* to protect
> the reproducibility of in-flight training runs. That split is a freeze artifact,
> **not a technical requirement**: TE 2.12 is compatible with the full-parameter
> path, and the two stacks are otherwise identical (same torch 2.11,
> transformers 5.9, megatron-core 0.16.0, mbridge). A fresh install should build
> **one** Megatron env on **TE 2.12 + `megatron-bridge`** (i.e.
> `requirements-lora.txt`) and use it for both full-FT and LoRA — see §3. We only
> keep a frozen TE-2.6 `.venv` around to reproduce the exact paper-run
> checkpoints.

---

## 1. FSDP / HuggingFace-Trainer reranker training — `.venv`

The baseline reranker training path (HF Trainer over PyTorch FSDP1 / DDP), and
also the home of Megatron full-parameter training and SPLADE encoding. Built
from `requirements-megatron.txt` (frozen 2026-05-21).

Core pins: `torch==2.11.0+cu128`, `transformers==5.9.0`, `accelerate==1.13.0`,
`peft==0.19.1`, `datasets==4.8.5`, `numpy==1.26.4`.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python packaging ninja setuptools wheel
uv pip install --python .venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
# flash-attn is a CUDA source build (needs a GPU node):
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 \
    uv pip install --python .venv/bin/python "flash-attn==2.8.1" --no-build-isolation --no-cache
uv pip install --python .venv/bin/python -r requirements-megatron.txt
uv pip install --python .venv/bin/python -e . --no-deps
```

> The Megatron stack (TransformerEngine, Megatron-Core, mbridge) is part of
> `requirements-megatron.txt` and is a heavy CUDA build. **For FSDP-only reranker
> training you do not need it** — the HF Trainer path uses just torch + flash-attn
> + transformers + accelerate. If you only want FSDP training, install those and
> `pip install -e . --no-deps`, and skip the `transformer-engine`/`megatron-core`/
> `mbridge` lines. For the full Megatron build, see §3.

DeepSpeed is intentionally **not** installed: ZeRO-3 has a known
gradient-gathering bug on this path, so the full-shard path uses FSDP1
`full_shard` instead (see `docs/bugs-and-fixes.md`). The FSDP config lives in
`deepspeed/fsdp_config_qwen3.json`.

---

## 2. Dense-retriever evaluation (faiss + pyserini) — **conda** `retriever`

Dense first-stage retrieval and lexical baselines need **faiss-gpu**, **pyserini**
(Lucene/BM25), and **BEIR**. This environment is **conda**, not uv/pip, for one
hard reason:

> **faiss-gpu does not install reliably from pip/uv.** The PyPI `faiss-gpu`
> wheels are stale/unofficial and routinely mismatch the CUDA runtime; the
> supported distribution is the conda-forge / pytorch channel build. We therefore
> keep dense-retrieval eval in a conda env (`/path/to/conda_envs/retriever`,
> `faiss==1.8.0`, 8 GPUs visible) rather than fighting pip.

Rebuild from the frozen export (preferred):

```bash
conda env create -p /path/to/conda_envs/retriever -f environment-retriever.yml
```

Or from scratch, if you need a clean build:

```bash
conda create -p /path/to/conda_envs/retriever python=3.10 -y
conda install -p /path/to/conda_envs/retriever -c pytorch -c nvidia faiss-gpu=1.8.0 -y
/path/to/conda_envs/retriever/bin/pip install pyserini==1.2.0 beir==2.2.0 pytrec_eval openai tiktoken
/path/to/conda_envs/retriever/bin/pip install "numpy==1.26.4"   # re-pin if bumped
```

**pyserini has three non-obvious gotchas on this cluster** (full notes in
`docs/bugs-and-fixes.md`); set these before any pyserini/BM25 run:

```bash
# 1. JDK 21+ (system Java is 11; pyserini needs jdk.incubator.vector)
conda install --override-channels -c conda-forge -p /path/to/conda_envs/retriever openjdk=21
export JAVA_HOME=/path/to/conda_envs/retriever/lib/jvm
# 2. libjvm.so is in a non-standard location for conda-forge openjdk
export JVM_PATH=/path/to/conda_envs/retriever/lib/jvm/lib/server/libjvm.so
export PATH=$JAVA_HOME/bin:$PATH
# 3. pyserini instantiates an OpenAI client at import -> needs a (dummy) key even for BM25
export OPENAI_API_KEY=dummy
```

SPLADE **learned-sparse** retrieval is different: it uses **Seismic**
(`pyseismic-lsr`, pure-Python, no JVM) and lives in `.venv`, not here. Installing
seismic re-pins numpy to 2.x — re-pin back to 1.26.4 afterward.

---

## 3. Megatron reranker training — one env (full-FT + LoRA)

The Megatron-Core backend (tensor/pipeline/expert parallelism, ZeRO-1
distributed optimizer) is the heaviest build: it compiles **TransformerEngine**
and **flash-attn** from source and pulls **Megatron-Core** + **mbridge** from
pinned git commits.

**Full build instructions are in
[`docs/megatron_installation.md`](megatron_installation.md)** — follow it rather
than improvising; install order matters and the CUDA-arch env vars are
package-specific:

| Package | Arch env var | Value (H200/H100) |
|---|---|---|
| torch / flash-attn | `TORCH_CUDA_ARCH_LIST` | `9.0` (with dot) |
| TransformerEngine | `NVTE_CUDA_ARCHS` | `90` (no dot) |

**A fresh install needs only one Megatron environment, for both full-parameter
and LoRA training.** Build it on **TE 2.12 + `megatron-bridge`** from
`requirements-lora.txt` — TE 2.12 runs the full-FT path fine, and
`megatron-bridge` adds the TP/EP-aware LoRA path on top. The scripted recipe
(GPU node, ~25–40 min, dominated by the TE compile):

```bash
bash scripts/setup_venv_lora.sh    # builds a TE-2.12 + megatron-bridge env
```

LoRA checkpoints are saved **pre-merged** (megatron-bridge), so they load as
ordinary HF models — no adapter-merge step, unlike the HF-PEFT path.

> Historically this was split into `.venv` (TE 2.6, full-FT) and `.venv-lora`
> (TE 2.12, LoRA) — see the note in "At a glance." That split was only to keep a
> TE-2.6 env frozen for in-flight-run reproducibility; it is not required for a
> fresh setup. If you must reproduce the exact paper-run full-FT checkpoints
> bit-for-bit, build the frozen TE-2.6 `.venv` from `requirements-megatron.txt`
> as well; otherwise the single TE-2.12 env above suffices.

---

## 4. Reranking & teacher annotation (vLLM) — `.venv-eval`

vLLM-based reranking (the high-throughput eval path) and offline teacher
annotation for distillation (`tevatron.utils.annotate_with_teacher`). Separate
because **vLLM pins a different torch than training**:
`vllm==0.19.1` on `torch==2.10.0`, vs `.venv`'s `torch==2.11.0` — ABI-isolated.

```bash
uv venv .venv-eval --python 3.12
uv pip install --python .venv-eval/bin/python -r requirements-eval.txt
uv pip install --python .venv-eval/bin/python -e . --no-deps
```

(numpy here is 2.x by design — vLLM tolerates it; the 1.26.4 pin only applies to
the training/seismic `.venv`.)

BEIR scoring (`tevatron.eval.run` / single-dataset `tevatron.eval.metrics`) and the HTTP `/score` serving pool
(`tevatron.eval.serve.server --backend vllm`) run here. The exact-match HF
scoring backend (`--backend hf`) runs in `.venv`.

---

## 5. LACONIC legacy cross-check — `.venv-laconic` (optional)

A transformers-4.51.3 environment used **only** to verify the migrated
decoder-LM SPLADE against the legacy `llm2vec` stack. **Not required** to run the
migrated code — `SpladeModelForCausalLM` runs on the main `.venv` (transformers
5.x). Built ad hoc:

```bash
uv venv .venv-laconic --python 3.12
uv pip install --python .venv-laconic/bin/python -r requirements-laconic.txt
uv pip install --python .venv-laconic/bin/python -e . --no-deps
```

See `examples/laconic/README.md` for what this env was used to establish (the
BOS attention-sink finding).

---

## Which environment for which command?

| Command | Env |
|---|---|
| `tevatron.retriever.driver.train` (dense/FSDP) | `.venv` |
| `tevatron.megatron.driver.train` / `distill_train` (full-FT **and** `--use_lora`) | `.venv` (single Megatron env; see §3) |
| `tevatron.retriever.driver.encode_splade` | `.venv` |
| `tevatron.retriever.driver.search_splade` (seismic) | `.venv` |
| dense retrieval / `pyserini` BM25 / faiss | conda `retriever` |
| `tevatron.utils.annotate_with_teacher` (vLLM teacher) | `.venv-eval` |
| `tevatron.eval.run` (sweep) / `tevatron.eval.metrics` (one dataset) | `.venv-eval` |
| `tevatron.eval.serve.server --backend vllm` | `.venv-eval` |
