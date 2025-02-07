# Inference RepLLaMA with VLLM

First install vllm `pip install vllm`

Since vllm currently not support lora load for embedding models (see [issue](https://github.com/vllm-project/vllm/issues/12808)), we first run the following python code to merge the lora weights into the model.
```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model('castorini/repllama-v1-7b-lora-passage')
model.half()
model.save_pretrained('repllama_merged')
tokenizer.save_pretrained('repllama_merged')
```
For the following steps, we use SciFact in BEIR as example.

### Encode corpus
```bash
mkdir beir_embedding_scifact
for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.vllm_encode  \
  --output_dir=temp \
  --model_name_or_path ./repllama_merged \
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
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.vllm_encode  \
  --output_dir=temp \
  --model_name_or_path ./repllama_merged \
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

# recall_100              all     0.9567
# ndcg_cut_10             all     0.7565
```

### Efficiency
Compared to the transformer implementation of encoding in [example_repllama.md](example_repllama.md), 
 which takes 1 minute 50 seconds to encode the entire SciFact corpus on a single H100 GPU, vLLM encoding only takes 45 seconds, making it 2.4 times faster.

### Citation
```bibtex
@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
```