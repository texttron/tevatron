# Retrieval

## Basic retrieval

Search is implemented with FAISS via [`tevatron.retriever.driver.search`](../src/tevatron/retriever/driver/search.py) (flat index, brute-force inner product on the stacked passage vectors).

Assume embeddings from encoding:

- query: `query_emb.pkl`
- corpus (possibly multiple shards): `corpus_emb00.pkl`, …, `corpus_emb19.pkl`

```bash
python -m tevatron.retriever.driver.search \
  --query_reps query_emb.pkl \
  --passage_reps 'corpus_emb*.pkl' \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to rank.txt
```

Corpus files are resolved with glob matching on `--passage_reps`. `--batch_size` is the number of queries per FAISS call; `-1` runs all queries in one batch (often fastest). With `--save_text`, each output line is `qid`, `pid`, `score` separated by tabs. Without `--save_text`, rankings are written as a pickle of `(scores, indices)`.

## Sharded search

If memory is tight, run search per shard and merge rankings.

```bash
INTERMEDIATE_DIR=intermediate
mkdir -p "${INTERMEDIATE_DIR}"
for s in $(seq 0 19)
do
python -m tevatron.retriever.driver.search \
  --query_reps query_emb.pkl \
  --passage_reps corpus_emb$(printf '%02d' $s).pkl \
  --depth 100 \
  --save_text \
  --save_ranking_to "${INTERMEDIATE_DIR}/$(printf '%02d' $s).txt"
done
```

Merge shard outputs with [`scripts/reduce_results.py`](../scripts/reduce_results.py):

```bash
python scripts/reduce_results.py \
  --results_dir "${INTERMEDIATE_DIR}" \
  --output rank.txt \
  --depth 100
```

`--depth` here caps how many documents per query are kept after merging scores across shards.
