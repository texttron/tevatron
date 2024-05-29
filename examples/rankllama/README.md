# RankLLaMA reranking example

> code tested on transformers==4.41.0 peft==0.11.1 A6000 GPU

In this example, we take the retrieval results from the first stage retriever RepLLaMA and do rerank with cross-encoder on RankLLaMA Passage Ranking datasets.

### Download run files from retriever
```
wget https://www.dropbox.com/scl/fi/xkhy3snts7bixzdqeg4ve/run.repllama.psg.dev.txt?rlkey=nzo68wbyj0xmsjvr0s5d32ano -O run.repllama.psg.dev.txt
wget https://www.dropbox.com/scl/fi/byty1lk2um36imz0788yd/run.repllama.psg.dl19.txt?rlkey=615ootx2mia42cxdilp4tvqzh -O run.repllama.psg.dl19.txt
wget https://www.dropbox.com/scl/fi/drgg9vj8mxe3qwayggj9o/run.repllama.psg.dl20.txt?rlkey=22quuq5wzvn6ip0c5ml6ad5cs -O run.repllama.psg.dl20.txt
```

For the following steps, we use TREC DL19 as example.
### Prepare reranker input files
```
python prepare_rerank_file.py \
    --query_data_name Tevatron/msmarco-passage \
    --query_data_split dl19 \
    --corpus_data_name Tevatron/msmarco-passage-corpus \
    --retrieval_results run.repllama.psg.dl19.txt \
    --output_path rerank_input.repllama.psg.dl19.jsonl \
    --depth 200
```

### Run Reranking
```
CUDA_VISIBLE_DEVICES=0 python -m tevatron.reranker.driver.rerank \
  --output_dir=temp \
  --model_name_or_path castorini/rankllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --dataset_path rerank_input.repllama.psg.dl19.jsonl \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --rerank_max_len $(( 32 + 164 )) \
  --dataset_name json \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --rerank_output_path run.rankllama.psg.dl19.txt
```

### Convert run format to trec
```
python -m tevatron.utils.format.convert_result_to_trec \
              --input run.rankllama.psg.dl19.txt \
              --output run.rankllama.psg.dl19.trec
```

### Run evaluation with pyserini

> install pyserini:
`pip install pyserini`

```
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage run.rankllama.psg.dl19.trec

Results:
ndcg_cut_10             all     0.7655
```


# Train Rank LLaMA from scratch
```
deepspeed --master_port 60000 --num_nodes 1 --num_gpus 1 --module tevatron.reranker.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir model_rankmistral \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --bf16 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --rerank_max_len $(( 32 + 164 )) \
  --num_train_epochs 1 \
  --logging_steps 100 \
  --gradient_accumulation_steps 4
```


### Citation
```
@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
```
