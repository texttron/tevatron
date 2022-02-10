# Retrieval

## Basic Retrieval
Tevatron implement the retrieval module based on FAISS.
Currently, the search is conducted by brute force search through the index.

Assume we already have query and corpus embeddings from encoding stage:

- query embedding: `query_emb.pkl`
- corpus embedding: `corpus_emb1.pkl, corpus_emb2.pkl, ..., corpus_emb19.pkl`

Call the `tevatron.faiss_retriever` module to retrieve,

```
python -m tevatron.faiss_retriever \  
--query_reps query.pkl \  
--passage_reps corpus_emb*.pkl \  
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to rank.txt
```

>Encoded corpus or corpus shards are loaded based on glob pattern matching of argument 
> `--passage_reps`. Argument `--batch_size` controls number of queries passed to the FAISS 
> index each search call and `-1` will pass all queries in one call. Larger batches typically
> run faster (due to better memory access patterns and hardware utilization.)
> Setting flag `--save_text` will save the ranking to a txt file with each line 
> being `qid pid score`.

## Sharded Search
As FAISS retrieval need to load corpus embeddings into memory, if the corpus embeddings are big,
we can alternatively paralleize search over the shards.
```bash
INTERMEDIATE_DIR=intermediate
mkdir ${INTERMEDIATE_DIR}
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.faiss_retriever \  
--query_reps query_emb.pkl \ 
--passage_reps corpus_emb${s}.pkl \
--depth 100 \
--save_ranking_to ${INTERMEDIATE_DIR}/${s}
done

```
Then combine the results using the reducer module,
```bash
python -m tevatron.faiss_retriever.reducer \
--score_dir ${INTERMEDIATE_DIR} \
--query query.pkl \
--save_ranking_to rank.txt
```
> Note: currently, `reducer` requires doc/query id being integer.