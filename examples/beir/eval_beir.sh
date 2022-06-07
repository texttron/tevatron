ckpt=$1
dataset=$2

mkdir beir_embeddings
for s in $(seq -f "%02g" 0 6)
do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${ckpt} \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --p_max_len 512 \
  --dataset_name Tevatron/beir-corpus:${dataset} \
  --encoded_save_path beir_embeddings/corpus_emb.${s}.pkl \
  --encode_num_shard 8 \
  --encode_shard_index ${s} &
done

for s in $(seq -f "%02g" 7 7)
do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${ckpt} \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --p_max_len 512 \
  --dataset_name Tevatron/beir-corpus:${dataset} \
  --encoded_save_path beir_embeddings/corpus_emb.${s}.pkl \
  --encode_num_shard 8 \
  --encode_shard_index ${s}
done

CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${ckpt} \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/beir:${dataset}/test \
  --encoded_save_path beir_embeddings/query.pkl \
  --q_max_len 512 \
  --encode_is_qry

sleep 30

set -f && OMP_NUM_THREADS=64 python -m tevatron.faiss_retriever \
    --query_reps beir_embeddings/query.pkl \
    --passage_reps beir_embeddings/corpus_emb.*.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to beir_embeddings/rank.txt

python -m tevatron.utils.format.convert_result_to_trec --input beir_embeddings/rank.txt --output beir_embeddings/rank.trec
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test beir_embeddings/rank.trec
