# `tevatron.eval` — retrieval/reranker evaluation

Score ranklists against qrels, rerank candidates, and aggregate a benchmark
sweep into one markdown table. The package separates three concerns so that
**adding a benchmark beyond BEIR-15 (MTEB, BRIGHT, a private set) is one registry
entry** — the metric core, scoring backends, and orchestrator are reused unchanged.

## Structure

```
eval/
  metrics.py      # INVARIANT core: read_ranklist() + score(qrels, results, k_values)
                  #   -> {ndcg, recall, map, mrr, precision}. No benchmark/model knowledge.
  benchmarks.py   # REGISTRY: BENCHMARKS["beir15"] = Benchmark(datasets, qrels loader,
                  #   split rule, summary layout). Add a benchmark = add an entry here.
  summary.py      # GENERIC rendering: SummaryLayout -> markdown table + aggregate rows.
  run.py          # ORCHESTRATOR CLI: per dataset resolve/rerank a ranklist -> metrics.score
                  #   -> cache JSON -> summary.render. Takes --benchmark (default beir15).
  rerank.py       # thin CLI: HF SeqCls reranker (DDP)         -> backends/hf.py
  rerank_vllm.py  # thin CLI: causal-LM yes/no reranker (vLLM) -> backends/vllm.py
  backends/
    hf.py         # AutoModelForSequenceClassification(num_labels=1), score = logits[:,0] at EOS
    vllm.py       # causal-LM yes/no: score = logprob(' yes') - logprob(' no')
    prompt.py     # RERANKER_PROMPT_SUFFIX + build_prompt() — train/eval source of truth
    score.py      # resolve_yes_no_token_ids() — the ' yes'/' no' continuation ids
    templates.py  # PromptTemplate registry: {tevatron, qwen3_reranker} + seqcls builder
  serve/          # persistent /score HTTP server (server.py) + pool client (client.py) + protocol.py
```

Two generic helpers that used to live here now sit in `tevatron.utils` (no
`tevatron` deps; `prepare_rerank_data` is also used by the retriever search
driver):

- `tevatron.utils.prepare_queries` — materialize a BEIR `queries.jsonl`.
- `tevatron.utils.prepare_rerank_data` — join a ranklist with corpus text into a rerank `jsonl`.

## The three layers, concretely

- **`metrics.score(qrels, results, k_values)`** is the only place metrics are
  computed (pytrec_eval via BEIR). It never knows what dataset it's scoring.
- **`benchmarks.Benchmark`** bundles the dataset list, a `load_qrels(dataset,
  cache_dir)` callable, `split_for(dataset)`, and a `SummaryLayout`. Its
  `.evaluate(dataset, ranklist, ...)` resolves qrels and delegates the math to
  `metrics.score`, returning a dict with a small provenance header
  (`dataset`/`split`/counts) — the schema cached as `<dataset>.metrics.json`.
- **`summary.render(results, layout, name, columns)`** turns per-dataset metrics
  into a table. *Which* datasets collapse into a sub-average (e.g. cqadupstack)
  and how the headline averages compose come from the `SummaryLayout`, not hardcode.

`BEIR-15` = BEIR-13 + msmarco-dev + cqadupstack (averaged over its 12 sub-forums)
= 15 task units. The registry stores all 26 underlying datasets; cqadupstack
collapses to one averaged unit in the table.

## Examples

> Scoring/aggregation needs BEIR (`beir`, `pytrec_eval`) — run in the eval env
> (`.venv-eval`) or the conda `retriever` env. pyserini-adjacent runs also need
> `OPENAI_API_KEY=dummy`. See `docs/environments.md`.

### 1. Score a sweep from existing ranklists (no model load)

First-stage runs or already-reranked outputs, resolved from a `{dataset}` pattern:

```bash
python -m tevatron.eval.run \
    --benchmark beir15 \
    --ranklist_pattern /path/to/eval_cache/e5_base/{dataset}/rank.text \
    --results_dir /path/to/eval_cache/results/e5_base --name e5-base
# writes <results_dir>/<dataset>.metrics.json (re-run skips cached) + summary.md
```

Score a single dataset (the old `beir_eval` entry point):

```bash
python -m tevatron.eval.metrics \
    --dataset scifact \
    --ranklist /path/to/scifact.rerank.text \
    --output  /path/to/scifact.metrics.json
```

Re-aggregate an existing results dir into a table (the old `aggregate` entry point):

```bash
python -m tevatron.eval.summary \
    --results_dir /path/to/eval_cache/results/e5_base \
    --output      /path/to/eval_cache/results/e5_base/summary.md
```

### 2. Rerank then score, one checkpoint, one dataset

HF sequence-classification reranker (DDP across visible GPUs):

```bash
torchrun --nproc_per_node=8 -m tevatron.eval.rerank \
    --model_name_or_path output/qwen3-0.6b-reranker \
    --rerank_input  /path/to/eval_cache/e5_base/scifact/rerank.jsonl \
    --rerank_output /path/to/eval_cache/results/myckpt/scifact.rerank.text \
    --rerank_max_len 512 --per_device_eval_batch_size 16 --append_eos_token

python -m tevatron.eval.metrics --dataset scifact \
    --ranklist /path/to/eval_cache/results/myckpt/scifact.rerank.text \
    --output   /path/to/eval_cache/results/myckpt/scifact.metrics.json
```

Causal-LM yes/no reranker via vLLM (single process; `tensor_parallel_size`, NOT torchrun):

```bash
python -m tevatron.eval.rerank_vllm \
    --model_name_or_path output/qwen3-0.6b-reranker/step_10136 \
    --rerank_input  /path/to/eval_cache/e5_base/scifact/rerank.jsonl \
    --rerank_output /path/to/eval_cache/results/myckpt-vllm/scifact.rerank.text \
    --rerank_max_len 512 --tensor_parallel_size 8 --prompt_template tevatron
```

The reranker input `jsonl` is produced from a first-stage ranklist + corpus:

```bash
python -m tevatron.utils.prepare_rerank_data \
    --dataset scifact \
    --rank_file   /path/to/eval_cache/e5_base/scifact/rank.text \
    --output_path /path/to/eval_cache/e5_base/scifact/rerank.jsonl --depth 100
```

### 3. Rerank + score a full sweep over a persistent backend pool

For a whole benchmark, load the model **once** behind an HTTP `/score` server and
drive the sweep as a thin client — amortizing load + graph capture across all
datasets. Backends may be local or remote (`ip:port`), one per GPU set.

```bash
# Host (a GPU node): one model load, served over HTTP. Repeat with disjoint
# CUDA_VISIBLE_DEVICES + distinct --port to put several backends on one node.
python -m tevatron.eval.serve.server --backend vllm \
    --model output/qwen3-8b-reranker --tensor_parallel_size 2 --port 8100
#   --backend hf --score_mode seqcls   # for HF SeqCls checkpoints

# Client (anywhere): rerank each dataset's rerank.jsonl through the pool, then score.
python -m tevatron.eval.run \
    --benchmark beir15 \
    --backends http://node:8100 http://node:8101 \
    --rerank_input_pattern /path/to/eval_cache/e5_base/{dataset}/rerank.jsonl \
    --results_dir /path/to/eval_cache/results/qwen3-8b --name qwen3-8b
```

## Adding a new benchmark

Add one entry to `BENCHMARKS` in `benchmarks.py`:

```python
BENCHMARKS["mybench"] = Benchmark(
    name="MyBench",
    datasets=[...],                       # ordered task list
    load_qrels=my_qrels_loader,           # (dataset, cache_dir) -> {qid: {pid: rel}}
    split_for=lambda d: "test",
    summary_layout=SummaryLayout(rows=[...], aggregates=[...]),
)
```

Then `python -m tevatron.eval.run --benchmark mybench ...` works immediately —
nothing in `metrics`, `summary`, `run`, or `backends` changes.

## Notes

- **Eval-only import hygiene.** `tevatron.eval.*` must not transitively import
  `megatron-core` (it would break the eval-only venv). The yes/no prompt suffix
  lives in `eval.backends.prompt`; `tevatron.megatron.data` re-exports it from
  there. Keep new `eval` code free of `tevatron.megatron` imports.
- **Two HF scoring contracts.** `--score_mode seqcls` (single regression logit at
  EOS, matches `tevatron.reranker` training) vs `yesno` (causal-LM ' yes'/' no'
  log-odds, matches the Megatron reranker). Loading the wrong head is silent —
  see `docs/bugs-and-fixes.md`.
- **Per-dataset caching.** `run.py` skips a dataset whose `<dataset>.metrics.json`
  already carries every requested column; pass `--overwrite` to re-score.
