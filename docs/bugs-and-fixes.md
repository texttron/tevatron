# Bugs & Fixes

Running log of non-obvious breakages hit while building the reranker training /
eval pipeline, and how they were resolved. Newest first.

Reference env (the stack these notes assume):
- `.venv` (uv-managed): `torch==2.11.0+cu128`, `transformers==5.9.0`,
  `accelerate==1.13.0`, `megatron-core`, tevatron editable.
- `.venv-lora`: megatron-bridge LoRA path.
- `.venv-eval`: `vllm==0.19.1` + `torch==2.10.0` (ABI-isolated from `.venv`).
- retriever conda env (`/path/to/conda_envs/retriever`): pyserini, faiss,
  beir, pyseismic-lsr.

---

## Decoder-LM SPLADE (LACONIC) collapses to identical vectors — the BOS attention sink

**Symptom.** Encoding the public LACONIC checkpoint (`utahnlp/laconic-1b`,
decoder-LM SPLADE) through the v3 `encode_splade` path gave garbage: scifact
NDCG@10 = 0.02 (paper 0.756). Every document and query produced near-identical
sparse vectors — pairwise cosine ≈ 0.99 — all dominated by the same generic
tokens (`Question`, `#`, `def`, `The`, `import`) at weight ~2.5, with content
tokens (`France`, `Paris`) at exactly 0.

**Red herring — the transformers version.** A GitHub issue on the legacy repo
reported the same degradation and "fixed" it by downgrading to transformers
4.51.3. We reproduced LACONIC's bidirectional path on **both** 4.51.3 (llm2vec
`_update_causal_mask` override) and 5.9 (`create_bidirectional_mask`) and got the
**identical** garbage. So it is NOT a version issue, and NOT the bidirectional
masking (we confirmed bidirectional is active three ways: causal-vs-bidir hidden
diff > 2.0, `is_causal=False` on all attention modules, eager honoring a `None`
mask as full attention). The transformers refactor (≤4.51 method
`_update_causal_mask` → 5.x module-level `create_causal_mask` /
`create_bidirectional_mask`) is real but orthogonal to this bug.

**Root cause — the `<|begin_of_text|>` (BOS) token is an attention sink.** Its
hidden state projects very large logits (~13) onto generic vocabulary tokens,
independent of the input. SPLADE max-pools `log(1+relu(logit))` over the
sequence, so that single BOS position wins the max for those generic tokens in
*every* sequence → all representations collapse onto the same basis. We traced
this directly: for the top pooled tokens, `argmax` over sequence position was
always position 0 (the BOS token).

The legacy LACONIC encode path tokenized with `prepare_for_model` on
pre-encoded ids with `add_special_tokens=False` (no BOS). The v3 collators
hard-coded `add_special_tokens=True`, prepending BOS — hence the regression.
Confirmed by toggling it: with BOS, scifact = 0.02 and doc-doc cosine 0.99;
without BOS, scifact = 0.648 and doc-doc cosine 0.09 (content tokens like
`brain`/`MRI`/`France` now lead).

**Second, smaller lever — the E5 prefixes.** LACONIC trains/encodes with
`query: ` / `passage: ` prefixes. Adding them on top of the BOS fix took scifact
0.648 → **0.752** (paper 0.756; residual is Seismic approximate-ANN
nondeterminism).

**Fix.** Added a backward-compatible `--add_special_tokens` DataArgument
(default `True`, so dense/MLM paths are unchanged) threaded through `TrainCollator`
and `EncodeCollator`. Decoder-LM SPLADE must run with `--add_special_tokens False`
plus `--query_prefix "query: " --passage_prefix "passage: "`. See
`examples/laconic/` for the validated recipe and the measured-effect table.

**Lesson.** For pooled (especially max-pooled) decoder-LM representations, the
attention-sink tokens (BOS, and sometimes EOS) are not neutral — a single sink
position can dominate a max-pool and wash out content. When a sparse/dense
encoder produces input-independent representations, check the *tokenization
contract* (special tokens, prefixes) before suspecting the model or the
framework version. The "downgrade transformers" folk fix here was coincidental.

---

## Why ZeRO-2/3 (Megatron-FSDP) is hard to support — and why we DON'T (ZeRO-1 only)

**Decision: the Megatron backend ships ZeRO-1 only** (distributed optimizer;
params/grads replicated). Param/gradient sharding, when needed, comes from tensor
parallelism. We attempted FSDP-style ZeRO-2/3 and abandoned it; this entry records
why, so nobody burns another day on it without new information.

**The attempt.** We added `--use_megatron_fsdp --dp_sharding_strategy
optim_grads{,_params}` flags and wired them into `engine.py`. It hit a *cascade*
of version-coupling failures, each unmasking the next:

1. **`use_megatron_fsdp` in the ddp_config alone does nothing.** mcore's standard
   `DistributedDataParallel` always builds the ZeRO-1 `_ParamAndGradBuffer`; it
   does not branch to the FSDP adapter on that flag. So the model stayed
   ZeRO-1-wrapped while the *optimizer* took its FSDP branch (it reads
   `model_chunk.ddp_config.use_megatron_fsdp`) → at the first step:
   `AttributeError: 'DistributedDataParallel' object has no attribute
   'param_and_grad_buffer'`.
2. **FSDP is incompatible with the distributed optimizer.** Megatron-FSDP shards
   optimizer state itself; `use_distributed_optimizer=True` assumes the standard
   DDP buffer layout and collides. (So FSDP forces the distributed optimizer
   off — one more coupled flag.)
3. **The wrapper is selected by mbridge, not by us, via a *dead* argument.**
   `mbridge.get_model` picks the DP wrapper from its own `use_torch_fsdp2` /
   `use_custom_fsdp` *arguments* — not the ddp_config. Setting `use_custom_fsdp=True`
   (the documented alias for megatron-FSDP) then failed with
   `ModuleNotFoundError: No module named 'megatron.core.distributed.custom_fsdp'`
   — **that module was removed in our mcore version.** The live impl is now
   `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` (`MCoreFSDP`), but
   mbridge's argument surface has no path to it. The only sharded wrapper mbridge
   can still reach is Torch FSDP2 (`use_torch_fsdp2`), whose `reshard_after_forward`
   knob does not cleanly map onto the ZeRO-2 vs ZeRO-3 distinction we wanted.

**Why it's not worth it.** Correct ZeRO-2/3 here requires three libraries to
agree (tevatron engine ↔ mbridge wrapper selection ↔ mcore FSDP module layout),
and the seam between them moves: `custom_fsdp` → `use_megatron_fsdp` rename,
module relocation, mbridge lagging the rename. Even a working configuration today
is fragile to the next mbridge/mcore bump. For a reranker toolkit it is not worth
chasing: **ZeRO-1 + TP/EP already fits everything we train** (dense 8B with TP=2;
30B-A3B MoE with EP=16), so FSDP-style param/grad sharding buys nothing we need.

**If revisited later**, the viable route is Torch FSDP2 via mbridge's
`use_torch_fsdp2=True` (it imports and exists), accepting `reshard_after_forward`
semantics instead of the `optim_grads`/`optim_grads_params` ladder — OR wrapping
`MCoreFSDP` directly and bypassing mbridge's wrapper selection. Both are real
engineering efforts, gated on a stable mbridge/mcore pairing. See the framework
roadmap in `megatron_reranker_usage.md`.

**Lesson.** When a feature spans three independently-versioned libraries, a
green import is not evidence it's wired correctly — trace the object that's
actually constructed (here: was the model wrapped in `MCoreFSDP` or plain DDP?).
And smoke-test before committing full runs; we caught each layer in a 2-minute
job, not a multi-hour one.

---

## Heads-up: Megatron backend supports ZeRO-1 (default) and ZeRO-2/3 (Megatron-FSDP)

Not a bug — a framing precision the paper must get right. The Megatron engine
(`src/tevatron/megatron/engine.py`) supports the full ZeRO ladder:

- **Default (no flag): ZeRO-1.** The *distributed optimizer*
  (`use_distributed_optimizer=True`) shards optimizer states (Adam m/v + fp32
  master) across DP ranks; **parameters and gradients are replicated**
  (`data_parallel_sharding_strategy='no_shard'`). Lightest-comm; the recommended
  default when the model fits (dense 8B + TP=2, or any model where TP/EP relieve
  memory).
- **`--use_megatron_fsdp --dp_sharding_strategy optim_grads`: ZeRO-2** (grads + opt
  sharded).
- **`--use_megatron_fsdp --dp_sharding_strategy optim_grads_params`: ZeRO-3**
  (params + grads + opt sharded, params all-gathered per layer — like HF FSDP
  `full_shard`).

Mapping: ZeRO-1 = shard optimizer only; ZeRO-2 = + grads; ZeRO-3 = + params.
(See the wiring caveat above for why FSDP turns the distributed optimizer off.)

**Why the *default* matters for the headline comparison.** The released
checkpoints + headline RQ1 wall-clock use **ZeRO-1 (+ light TP)**, so that
comparison is **Megatron ZeRO-1 vs HF FSDP `full_shard` = ZeRO-3** — *different
sharding degrees*, not just different frameworks. We control for this two ways:
(1) an HF `shard_grad_op` (ZeRO-2) run, the closest HF offers, showed HF ZeRO-2 ≈
ZeRO-3 in speed, so the sharding stage is not what makes HF slow; and (2) a
matched **Megatron TP=1/DP=8 + ZeRO-1/2/3** sweep isolates the DP scheme on the
Megatron side at the same DP=8 as HF FSDP. State the config precisely rather than
calling both "sharded DP."

For MoE, EP shards the experts, so ZeRO-1 on top of EP=16 already fits 30B-A3B —
the "EP is the unlock, not FSDP-style sharding" point for RQ2.

Peak memory is recoverable from each run's **wandb system metrics**
(`gpu.*.memoryAllocatedBytes`, read from the local `.wandb` binary via the
datastore reader) — no `skip_memory_metrics` / `max_memory_allocated`
instrumentation needed.

---

## HF reranker is seq-cls (scores at EOS), not a yes/no causal-LM probe

**Symptom.** The EOS-fixed HF retrains (FSDP full-FT, DDP-LoRA) trained with a
perfectly normal loss curve, but reranking was garbage: scifact NDCG@10 = 0.29
(full-FT) vs 0.79 for the equivalent Megatron checkpoint. A toy probe showed
scores saturated at ~15 with an *irrelevant* doc ranked above a relevant one.

**Root cause — two architectures, two scoring positions, conflated.** The two
reranker training paths score differently:

- **Megatron path**: causal-LM yes/no scorer. Score = `logit(' yes') - logit(' no')`
  at the *prompt's final `?`*. Must NOT append EOS (that shifts the probe to
  `<eos>`, breaking train/eval alignment).
- **HF path** (`tevatron.reranker.modeling.RerankerModel`): it's
  `AutoModelForSequenceClassification(num_labels=1)` — a single regression-logit
  score read from the **last non-pad token**. Training MUST `--append_eos_token`
  so that scored position is well-defined and stable. The checkpoint stores a
  `score.weight` head and has **no `lm_head`**.

We earlier removed `--append_eos_token` from the HF scripts believing it was the
yes/no probe-position fix — but that fix only applies to the Megatron causal-LM
path. Dropping it from the seq-cls HF path moved the scored token off EOS,
so the trained `score` head was never exercised at a consistent position.

A second, compounding bug surfaced while diagnosing: the serve `HFBackend` loaded
these checkpoints with `AutoModelForCausalLM` and scored yes/no token logits.
That silently **discards the trained `score.weight` and uses a random `lm_head`**
→ saturated garbage. (Loading a seq-cls checkpoint as causal-LM does not error;
the head mismatch is silent.)

**Fix.** (1) Re-add `--append_eos_token` to both HF training scripts; the comment
now spells out the seq-cls-vs-causal-LM distinction. (2) The serve HFBackend must
load seq-cls checkpoints via `AutoModelForSequenceClassification(num_labels=1)`
and score `.logits[:,0]` at the EOS position — NOT the causal-LM yes/no contract.
Bad checkpoints deleted and retrained (jobs 20759 / 20760).

**Lesson.** "Loss is fine but eval is garbage" almost always means a
train/inference *contract* mismatch (scored position, head type, tokenization),
not a training failure. Check the saved keys (`score.weight` vs `lm_head.weight`)
to identify the head before choosing the serving Auto* class.

---

## vLLM emits garbage tokens for one specific checkpoint (HF path fine)

**Symptom.** The Megatron contrastive-LoRA 8B checkpoint scored NDCG@10 = 0.0235
on scifact via the vLLM rerank backend, but **0.8159 via the HF DDP backend** —
same checkpoint, same prompt, same yes/no token IDs. Every vLLM score was
*exactly* 0.0.

**Diagnosis (ruled out, then found).**
- All 60000 vLLM scores were 0.0 because `score = lp(' yes') - lp(' no')` and
  BOTH ` yes`/` no` fell outside vLLM's top-20 logprobs, so both hit the same
  `-20.0` sentinel and cancelled (`_backend_vllm.py`).
- Ruled out: tokenizer (identical), resolved yes/no IDs (both 9834/902),
  safetensors index (399 weights, intact, embed/lm_head/layers.35 present), load
  warnings (none), TP (probe at TP=1 *also* failed; original was TP=2).
- Hosted the checkpoint on a vLLM server and probed top-20 at the generation
  position: it emits ` defence`/` Rand`/` Lib`/` Ireland` — coherent English
  nouns, flat distribution (top logprob −2.0), **no yes/no anywhere**. The
  *working* distill-LoRA checkpoint on identical config emits ` Yes`/` yes`/
  ` No` (top −4.07). So the model weights are correct (HF gives 0.82) but
  vLLM computes a divergent forward for this one checkpoint's merged weights.

**Verdict.** A bf16 numerical edge case in vLLM's kernels for this specific
merged checkpoint, NOT a bad checkpoint, bad save, or scoring bug. Both LoRA
checkpoints were merged by megatron-bridge at save time (full 16GB weights), so
no runtime adapter merge is involved — earlier "vLLM LoRA-merge bug" framing was
wrong.

**Mitigation.** (a) The HF DDP rerank backend (`tevatron.megatron.eval.rerank`)
is exact-match to training math and unaffected — use it for headline quality
numbers; vLLM is for throughput. (b) Expected to disappear with a different data
seed (shuffles the exact merged weight values). (c) Caveat to note in the
writeup, not a blocker. Not worth chasing the vLLM kernel further given a working
path exists.

**Related (separate concern):** to keep HF-PEFT *adapter* checkpoints off vLLM's
runtime-LoRA path entirely, `tevatron.utils.merge_lora` pre-merges an adapter
into a plain bf16 model dir. (The Megatron LoRA path already saves merged.)

---

## HF Trainer FSDP: layered FSDP1/FSDP2 shim, flag churn, and save correctness

**Symptom(s).**
1. `ValueError: Some specified arguments are not used by the HfArgumentParser:
   ['--fsdp_transformer_layer_cls_to_wrap', 'Qwen3DecoderLayer']` — the
   standalone flag was removed; in transformers 5.x the wrap class lives inside
   `--fsdp_config` JSON (`transformer_layer_cls_to_wrap`).
2. Warning: "When using FSDP full shard, instead of `gradient_checkpointing` in
   TrainingArguments, use `activation_checkpointing` in `fsdp_config`. The
   former introduces a redundant AllGather in backward." (non-fatal, but real
   overhead — moved into the fsdp_config).

**Root cause.** HF Trainer's FSDP path is a compatibility shim over BOTH
`torch.distributed.fsdp.FullyShardedDataParallel` (FSDP1) and the newer
`fully_shard` (FSDP2), gated by accelerate version + config flags. Wrap policy,
mixed precision, and state-dict gathering each route through a different layer
(Trainer -> Accelerator -> FSDPPlugin -> torch), and the flag surface drifts
across releases. The moved-flag error above is that layering leaking.

**Fix applied (no training-loop rewrite).** Use a `--fsdp_config` JSON instead
of standalone flags: `deepspeed/fsdp_config_qwen3.json` with
`transformer_layer_cls_to_wrap: ["Qwen3DecoderLayer"]`,
`activation_checkpointing: true`, `use_orig_params: true`. Drop
`--gradient_checkpointing` from the CLI. Script:
`scripts/megatron_support/train_qwen3_8b_reranker_hf_fsdp.sh`.

**The non-obvious risk — checkpoint correctness, not crashes.** Under FSDP
`full_shard`, each rank holds only a shard. `RerankerTrainer._save` originally
called `self.model.save()` -> `save_pretrained()` directly, IGNORING the
consolidated `state_dict` that `Trainer.save_model()` gathers and passes in — so
it would have written this rank's shard, not the full model. No crash; just a
silently wrong checkpoint that evals badly. Fixed in
`src/tevatron/reranker/trainer.py` (`_save` now passes the gathered, de-prefixed
state_dict through to `save_pretrained`; LoRA path extracts the adapter). If a
future FSDP1/FSDP2 mismatch makes `get_state_dict()` return shards on non-zero
ranks, this would resurface — VERIFY a saved checkpoint loads + evals sanely.

**Workable path forward (not done).** This is solvable by pinning
transformers/accelerate to a known-good pair — trial-and-error, but avoids a
rewrite. The clean-but-heavier alternative is a standalone native FSDP2
(`fully_shard`) training loop with explicit
`get_model_state_dict(StateDictOptions(full_state_dict=True, cpu_offload=True))`
for saves (~40 legible lines vs the HF shim). Deferred: the HF path works once
the config JSON + `_save` fix are in.

**Design choice that sidesteps it for LoRA.** Only full-FT 8B actually needs
sharding (FSDP) — its AdamW model states (~128 GB) OOM a 141 GB H200 under plain
DDP. LoRA trains ~20M params, so the frozen base just replicates and plain DDP
fits (~30-40 GB). So: full-FT -> FSDP, LoRA -> DDP. Avoids the FSDP+PEFT wrap
interaction entirely.

---

## DeepSpeed not installed in `.venv`

**Symptom.** `ImportError: DeepSpeed is not available => install it using
pip3 install deepspeed` when launching the HF ZeRO-3 reranker run.

**Resolution.** Did NOT install it. Pivoted the full-shard path to native FSDP
(see above) — DeepSpeed ZeRO-3 also has a known grad-gather issue, and FSDP
`full_shard` is the equivalent strategy without adding a heavy dependency
(deepspeed's resolver also tried to bump numpy, which we keep pinned at 1.26.4).

---

## pyseismic-lsr install bumped numpy 1.26.4 -> 2.4

**Symptom.** `uv pip install pyseismic-lsr` silently upgraded numpy to 2.4.6,
which risks breaking the torch 2.11 training path (numpy 2.x ABI changes).

**Fix.** Re-pin after installing seismic deps:
`uv pip install --python .venv/bin/python "numpy==1.26.4"`. Verified seismic's
`SeismicIndexLV` build + search still work under numpy 1.26.4. Keep numpy at
1.26.4 in `.venv`.

---

## SeismicIndexLV is not picklable (index cache crash)

**Symptom.** `TypeError: cannot pickle 'builtins.SeismicIndexLV' object` —
crashed `search_splade.py` AFTER a 25-min index build over the 8.8M-doc msmarco
corpus (so the expensive work was wasted and no rank.text was written).

**Root cause.** `SeismicIndexLV` is a Rust-backed object; `pickle.dump` can't
serialize it. The `--index_cache` feature used pickle.

**Fix.** `src/tevatron/retriever/driver/search_splade.py` now uses seismic's
native `index.save(path)` / `SeismicIndexLV.load(path)` and wraps the whole
cache step in try/except so a cache failure can NEVER kill the (expensive)
search. Note seismic's `save(path)` takes a path WITHOUT extension and appends
`.index.seismic`; `load()` wants the full `.index.seismic` path — keep those
consistent if re-enabling the cache.

---

## tevatron collator/dataset hard-import qwen_omni_utils + PIL

**Symptom.** `ModuleNotFoundError: No module named 'qwen_omni_utils'` (then
`PIL`) when importing the SPLADE encode driver — these are multimodal-only deps
pulled in at module top of `tevatron.retriever.collator` / `.dataset`.

**Fix.** Made `process_mm_info` a lazy wrapper (import inside the function) in
`collator.py`; installed Pillow (tiny, no numpy disturbance) for `dataset.py`.
The text-only encode/retrieve path no longer needs the multimodal stack.
