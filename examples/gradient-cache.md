# Gradient Cache
Tevatron adopts gradient cache technique to allow large batch training of dense retriever on memory limited GPU.

Details is described in paper [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup
](https://arxiv.org/abs/2101.06983).
```
@inproceedings{gao2021scaling,
     title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
     author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
     booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
     year={2021},
}
```

## Install GradCache Package
Follow the instruction on the [GradCache repo](https://github.com/luyug/GradCache#installation).

## GradCache Flags
We provide flags:
- `--grad_cache`: enable gradient caching
- `--gc_q_chunk_size`: sub-batch size for query 
- `--gc_p_chunk_size`: sub-batch size for passage

Set `--gc_q_chunk_size` and `--gc_p_chunk_size` such that the correponding query/passage sub-batch can fit in GPU memory for gradient computation.

## Example: Training DPR on Natural Question
```bash
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --train_dir nq-train \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --negatives_x_device \
  --grad_cache \
  --gc_q_chunk_size $Q_CHUNK_SIZE \
  --gc_p_chunk_size $P_CHUNK_SIZE
```

It is recommended to set `Q_CHUNK_SIZE` and `P_CHUNK_SIZE` to be as large as GPU RAM can hold.

You can find full DPR example in [DPR example folder](dpr). 

