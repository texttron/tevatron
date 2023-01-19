ckpt=$1
dataset=$2
tokenizer=bert-base-uncased
embedding_dir=beir_embedding_${ckpt}

mkdir $embedding_dir
for s in $(seq -f "%02g" 0 7)
do
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${ckpt} \
  --tokenizer_name ${tokenizer} \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --p_max_len 512 \
  --dataset_name Tevatron/beir-corpus:${dataset} \
  --encoded_save_path $embedding_dir/corpus_${dataset}.${s}.pkl \
  --encode_num_shard 8 \
  --encode_shard_index ${s}
done

CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${ckpt} \
  --tokenizer_name ${tokenizer} \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/beir:${dataset}/test \
  --encoded_save_path $embedding_dir/query_${dataset}.pkl \
  --q_max_len 512 \
  --encode_is_qry

set -f && OMP_NUM_THREADS=12 python -m tevatron.faiss_retriever \
    --query_reps $embedding_dir/query_${dataset}.pkl \
    --passage_reps $embedding_dir/corpus_${dataset}.*.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $embedding_dir/rank.${dataset}.txt

python -m tevatron.utils.format.convert_result_to_trec --input $embedding_dir/rank.${dataset}.txt \
                                                       --output $embedding_dir/rank.${dataset}.trec \
                                                       --remove_query
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test $embedding_dir/rank.${dataset}.trec
