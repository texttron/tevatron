```
CUDA_VISIBLE_DEVICES=7 python distil_train.py \
  --output_dir student_msmarco \
  --model_name_or_path distilbert-base-uncased \
  --teacher_model_name_or_path reranker_msmarco \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 1e-5 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --overwrite_output_dir
```