# Condenser/coCondenser Natural Question Retrieval
## Condenser/coCondenser
You can find details about Condenser/coCondenser pre-training in papers ([Condenser](https://arxiv.org/abs/2104.08253); [coCondenser](https://arxiv.org/abs/2108.05540)) and [open source code](https://github.com/luyug/Condenser),
```
@inproceedings{gao-callan-2021-condenser,
    title = "Condenser: a Pre-training Architecture for Dense Retrieval",
    author = "Gao, Luyu and Callan, Jamie",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.75",
    pages = "981--993",
}
```
```
@misc{gao2021unsupervised,
      title={Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval}, 
      author={Luyu Gao and Jamie Callan},
      year={2021},
      eprint={2108.05540},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## Training with BM25 Negatives
Download bm25 training data from DPR server,
```
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gunzip biencoder-nq-train.json.gz
```
Run tokenization,
```
mkdir nq-train
python prepare_wiki_train.py --input biencoder-nq-train.json --output nq-train/bm25.bert.json --tokenizer bert-base-uncased
```
Optioinally, grab mined hard negatives from our server,
```
wget http://boston.lti.cs.cmu.edu/luyug/co-condenser/nq/hn.json.gz
gunzip hn.json.gz
python prepare_wiki_train.py --input hn.json --output nq-train/hn.bert.json --tokenizer bert-base-uncased
```
Pick a pre-trained condenser that is most suitable for the experiment from [Condenser Repo](https://github.com/luyug/Condenser#pre-trained-models).
Run training,
```
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model-nq \
  --model_name_or_path CONDENSER_MODEL_NAME \
  --do_train \
  --save_steps 20000 \
  --train_dir nq-train \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 5e-6 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 40 \
  --negatives_x_device \
  --untie_encoder \
  --positive_passage_no_shuffle
```

## Training with Hard Negatives
Download bm25 training data from DPR server,
```
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gunzip biencoder-nq-train.json.gz
```
Run tokenization,
```
mkdir nq-train
python prepare_wiki_train.py --input biencoder-nq-train.json --output nq-train/bm25.bert.json --tokenizer bert-base-uncased
```
In addition, grab mined hard negatives from our server,
```
wget http://boston.lti.cs.cmu.edu/luyug/co-condenser/nq/hn.json.gz
gunzip hn.json.gz
python prepare_wiki_train.py --input hn.json --output nq-train/hn.bert.json --tokenizer bert-base-uncased
```

Pick a pre-trained condenser that is most suitable for the experiment from [Condenser Repo](https://github.com/luyug/Condenser#pre-trained-models).
Run training,
```
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model-nq \
  --model_name_or_path CONDENSER_MODEL_NAME \
  --do_train \
  --save_steps 20000 \
  --train_dir nq-train \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 5e-6 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 20 \
  --negatives_x_device \
  --untie_encoder \
  --positive_passage_no_shuffle
```

## Encode
We will use NQ corpus hosted by Tevatron on the huggingface hub,

```
OUTDIR=temp
MODEL_DIR=nq-model

for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --config_name CONDENSER_MODEL_NAME \
  --output_dir=$OUTDIR \
  --model_name_or_path $MODEL_DIR \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --p_max_len 256 \
  --dataset_proc_num 8 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path embeddings-nq/$s.pt \
  --encode_num_shard 20 \
  --passage_field_separator sep_token \
  --encode_shard_index $s
done

python -m tevatron.driver.encode \
  --output_dir=$OUTDIR \
  --model_name_or_path $MODEL_DIR \
  --config_name CONDENSER_MODEL_NAME \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --q_max_len 32 \
  --dataset_proc_num 2 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path embeddings-nq-queries/query.pt \
  --encode_is_qry
```

## Search and Evaluation
### Search
```bash
ENCODE_QRY_DIR=embeddings-nq-queries
ENCODE_DIR=embeddings-nq
DEPTH=200
RUN=run.nq.test.txt
python -m tevatron.faiss_retriever \
--query_reps $ENCODE_QRY_DIR/query.pt \
--passage_reps $ENCODE_DIR/'*.pt' \
--depth $DEPTH \
--batch_size -1 \
--save_text \
--save_ranking_to run.nq.test.txt
```
Convert result to trec format
```
python -m tevatron.utils.format.convert_result_to_trec --input run.nq.test.txt --output run.nq.test.teIn
```
### Evaluation
Tevatron does not currently contain eval code path.
Evaluate with Pyserini for now, `pip install pyserini`

Recover query and passage contents
```
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run --topics dpr-nq-test \
                                                                --index wikipedia-dpr \
                                                                --input run.nq.test.teIn \
                                                                --output run.nq.test.json
```

```
python -m pyserini.eval.evaluate_dpr_retrieval --retrieval run.nq.test.json --topk 5 20 100
```
