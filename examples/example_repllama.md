# TevatronV2 RepLLaMA example

## Inference RepLLaMA
For the following steps, we use SciFact in BEIR as example.
### Encode corpus
```bash
mkdir beir_embedding_scifact
for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --passage_prefix "passage: " \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config scifact \
  --dataset_split train \
  --encode_output_path beir_embedding_scifact/corpus_scifact.${s}.pkl \
  --encode_num_shard 4 \
  --encode_shard_index ${s}
done
```

### Encode queries

```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --query_prefix "query: " \
  --append_eos_token \
  --query_max_len 512 \
  --dataset_name Tevatron/beir \
  --dataset_config scifact \
  --dataset_split test \
  --encode_output_path beir_embedding_scifact/queries_scifact.pkl \
  --encode_is_query
```

### Search
```bash
python -m tevatron.retriever.driver.search \
    --query_reps beir_embedding_scifact/queries_scifact.pkl \
    --passage_reps 'beir_embedding_scifact/corpus_scifact*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to beir_embedding_scifact/rank.scifact.txt
    
```


### Convert to TREC format
```bash
python -m tevatron.utils.format.convert_result_to_trec --input beir_embedding_scifact/rank.scifact.txt \
                                                       --output beir_embedding_scifact/rank.scifact.trec \
                                                       --remove_query
```

### Evaluate
```bash
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-scifact-test beir_embedding_scifact/rank.scifact.trec

# recall_100              all     0.9467
# ndcg_cut_10             all     0.7561
```

### Citation
```bibtex
@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
```
