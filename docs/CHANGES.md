# Tevatron 3.0 — Release Notes (v2 → v3)

Tevatron 3.0 adds three capabilities on top of v2. The guiding principle is
**additive and backward compatible**: new features plug into the existing data
path, collator, trainer, and HF-loadable checkpoint conventions, and the
dense / MLM / reranker paths behave exactly as before unless a new flag is set.

1. **Megatron reranker training backend** — tensor / pipeline / expert
   parallelism for reranker training at scale.
2. **Unified reranker evaluation** — one CLI + an HTTP serving pool over HF and
   vLLM backends, plus an offline distillation pipeline.
3. **Decoder-LM SPLADE (LACONIC)** — learned-sparse first-stage retrieval with a
   decoder LM backbone.

The three are independent (separate modules and drivers); use any subset.

---

## 1. Megatron training backend — `src/tevatron/megatron/`

A second reranker training backend alongside the HuggingFace-Trainer path,
exposed through the same CLI and consuming the same training data. Adds the
parallelism that PyTorch FSDP1 lacks: tensor (TP), pipeline (PP), and
**expert (EP)** parallelism, with a ZeRO-1 distributed optimizer — enabling
reranker training up to 30B-A3B MoE.

| File | Purpose |
|---|---|
| `megatron/engine.py` | Builds the Megatron-Core model via the HF↔Megatron weight bridge (mbridge full-FT / megatron-bridge LoRA); TP/PP/EP/DP config, distributed optimizer, activation recompute. |
| `megatron/config.py` | Backend config dataclass (parallel sizes, recompute, MoE/router knobs). |
| `megatron/data.py`, `distill_data.py` | Dataset adapters; reuse the existing Tevatron data path. |
| `megatron/loss.py` | Listwise CE (contrastive) and listwise-KL (distillation) over grouped yes/no scores. |
| `megatron/driver/train.py`, `distill_train.py` | Contrastive and distillation entry points. |
| `megatron/eval/` | HF-backend rerank scoring for headline quality numbers. |

**Key design points** (see `docs/megatron_reranker_usage.md`):
- Reranker scored as a plain causal LM (`logit(' yes') − logit(' no')` at the
  final prompt token) — no value head, runs identically under any parallelism.
- Checkpoints written in **HF format** via the bridge (LoRA pre-merged), so they
  load through the normal HF/vLLM path with no conversion.
- Data parallelism is **ZeRO-1** (distributed optimizer; params/grads
  replicated); param/grad sharding comes from TP/EP. See
  `docs/megatron_reranker_usage.md` for the roadmap.
- A named **LoRA target-group registry** (`attn`, `moe_experts`, `moe_shared`,
  `moe_router`) avoids the MoE foot-gun where a dense MLP pattern silently
  matches all 128 expert FFNs.

**Supporting files:** `examples/megatron/` (training recipes: contrastive
dense/MoE, LoRA, two-step distillation), `scripts/megatron_support/`
(ready-to-run train/eval scripts incl. SLURM multi-node wrappers),
`requirements-megatron.txt` / `requirements-lora.txt`, `docs/megatron_installation.md`.

---

## 2. Unified evaluation + distillation

### Evaluation — `src/tevatron/eval/`

The eval package separates three concerns so adding a benchmark beyond BEIR-15
(MTEB, BRIGHT, a private set) = **adding one registry entry**, with the metric
core, scoring backends, and orchestrator reused unchanged:

1. **Invariant metrics** (`metrics.py`) — `read_ranklist` + `score(qrels,
   results, k_values)` → NDCG/Recall/MAP/MRR/Precision. No benchmark/model knowledge.
2. **Benchmark registry** (`benchmarks.py`) — a `Benchmark` carries the dataset
   list, qrels loader, split convention, and summary layout. `BENCHMARKS["beir15"]`
   is BEIR-13 + msmarco-dev + cqadupstack (avg of 12) = 15 task units.
3. **Generic rendering** (`summary.py`) — `SummaryLayout`-driven markdown table;
   which datasets collapse into a sub-average is a benchmark fact, not hardcoded.

The orchestrator `run.py` (`--benchmark`, default `beir15`) ties them together
with two usage schemas behind one CLI:

- **Score-only (local):** score/aggregate existing ranklists, no model load.
- **Rerank-and-score over a remote backend pool (HTTP):** the reranker is hosted
  as a persistent `/score` server (`serve/server.py`, `_BaseBackend` with
  `HFBackend`/`VLLMBackend`); the evaluator (`serve/client.py`) load-balances
  candidate batches across one or more backend URLs.

| File | Purpose |
|---|---|
| `eval/metrics.py` | Invariant scoring core (read ranklist, score vs qrels). Also a single-dataset CLI. |
| `eval/benchmarks.py` | Benchmark registry: dataset lists, BEIR qrels loader, summary layout. |
| `eval/summary.py` | Generic markdown rendering from a `SummaryLayout`. Also an aggregate-from-dir CLI. |
| `eval/run.py` | Benchmark-agnostic sweep orchestrator (`--benchmark`). |
| `eval/rerank.py`, `rerank_vllm.py` | Thin rerank CLIs over the backends. |
| `eval/backends/{hf,vllm}.py` | HF SeqCls + vLLM yes/no scoring backends. |
| `eval/backends/{prompt,score,templates}.py` | Prompt construction, yes/no token resolution, template registry. |
| `eval/serve/` | FastAPI `/score` server, client, protocol — the HTTP schema. |
| `utils/prepare_queries.py`, `utils/prepare_rerank_data.py` | Query/rerank-input materialization. |

See `src/tevatron/eval/README.md` for the full pipeline and examples.

### Distillation — `src/tevatron/utils/annotate_with_teacher.py`

`tevatron.utils.annotate_with_teacher` precomputes per-passage teacher scores
**offline** into the dataset, so distillation training adds no teacher forward
pass at train time. It's a standalone `python -m` driver; consumed by
`megatron/driver/distill_train.py`.

---

## 3. Decoder-LM SPLADE — LACONIC

A decoder LM (Llama 3 / Qwen2.5) becomes a SPLADE-style first-stage retriever:
bidirectional attention → vocab logits via the tied `lm_head` → max-pool over
the sequence → sparse vector; trained contrastively with FLOPS sparsity
regularization; retrieved with Seismic.

### New files

| File | Purpose |
|---|---|
| `retriever/modeling/bidirectional.py` | **In-place** causal→bidirectional conversion using transformers 5.x's `create_bidirectional_mask` (swaps the trunk's mask builder, clears `is_causal`). Gated to validated `model_type`s (`llama`, `qwen2`). |
| `retriever/splade_trainer.py` | `SpladeTrainer` (subclasses `TevatronTrainer`) + FLOPS regularizer and quadratic warmup scheduler. Base trainer untouched. |
| `retriever/driver/train_splade.py` | Causal-SPLADE training entry point. |

### Modified existing files (backward compatible)

| File | Change |
|---|---|
| `retriever/modeling/splade.py` | Added `SpladeModelForCausalLM` (decoder backbone, max/mean/last pooling, optional bidirectional, top-k). The existing MLM `SpladeModel` is unchanged. |
| `retriever/modeling/__init__.py` | Export `SpladeModelForCausalLM`. |
| `retriever/arguments.py` | Added `is_bidirectional`, `pooling_strategy`, **`add_special_tokens`** (default `True`), and a `SpladeTrainingArguments` subclass with FLOPS knobs (`q/p_flops_loss_factor`, `flops_warmup`). |
| `retriever/collator.py` | Thread `add_special_tokens` through `TrainCollator` / `EncodeCollator` (was hardcoded `True`). |
| `retriever/driver/encode_splade.py` | Added `--splade_model_type {mlm,causal}`, `--splade_topk` (top-k term cap), causal-model branch with bidirectional/pooling. |
| `retriever/driver/search_splade.py` | Added native merged-file `SeismicIndexLV.build` fast path (large corpora) alongside the in-memory build. |
| `retriever/modeling/dense.py` | Made the Qwen2.5-Omni import **lazy** so the text-only path imports on transformers versions without the multimodal class. |

### Critical correctness finding (documented in `docs/bugs-and-fixes.md`)

Reproducing the public `utahnlp/laconic-1b` checkpoint surfaced a non-obvious
bug: with default tokenization the model produced **identical** sparse vectors
for every input (scifact NDCG@10 ≈ 0.02). Root cause — **the `<|begin_of_text|>`
(BOS) token is an attention sink** whose hidden state projects huge logits onto
generic tokens; SPLADE max-pool grabs that position for every sequence,
collapsing all representations. The legacy encode path never added BOS.

**Fix:** the `--add_special_tokens False` flag (above), plus E5-style
`--query_prefix "query: "` / `--passage_prefix "passage: "`. End-to-end on the
public checkpoint this reproduces **scifact NDCG@10 = 0.752** (paper 0.756);
progression 0.02 → 0.648 (no BOS) → 0.752 (+ prefixes). See
`examples/laconic/` for the validated recipe.

Echo embedding and in-model hard-top-k are intentionally **not** ported (rarely
used; not needed to reproduce the headline LACONIC results).

---

## Documentation

| Doc | Contents |
|---|---|
| `docs/environments.md` | **Start here for setup.** The separate environments, what each is for, why they're separate, and a per-command env table. |
| `docs/megatron_installation.md` | Installing the Megatron stack (TransformerEngine, Megatron-Core, mbridge). |
| `docs/megatron_reranker_usage.md` | Parallelism strategies, contrastive vs distillation, LoRA target groups, checkpoint format, roadmap. |
| `docs/bugs-and-fixes.md` | Non-obvious breakages + fixes (HF seq-cls scoring contract, the LACONIC BOS sink, env pins). |
| `src/tevatron/eval/README.md` | Evaluation pipeline structure + examples. |
| `examples/megatron/README.md` | Megatron reranker training recipes (contrastive dense/MoE, LoRA, distillation). |
| `examples/laconic/README.md` | LACONIC reproduction recipe. |

## Pinned dependency manifests (repo root)

Every environment has a frozen, version-pinned manifest so rebuilds are
reproducible; each file's header carries its exact rebuild command.

| File | Environment |
|---|---|
| `requirements-megatron.txt` | Training, TE 2.6 (frozen for paper-run reproduction) |
| `requirements-lora.txt` | Megatron env, TE 2.12 + `megatron-bridge` (full-FT + LoRA; preferred for fresh installs) |
| `requirements-eval.txt` | vLLM + BEIR + teacher annotation |
| `environment-retriever.yml` | conda — faiss-gpu + pyserini + BEIR |

See `docs/environments.md` for which env runs which command and why they're separate.
