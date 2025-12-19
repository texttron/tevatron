output_dir=retriever-qwen3-emb-ft-chunk-1219-1
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --bf16 \
  --per_device_eval_batch_size 4 \
  --normalize \
  --pooling last \
  --padding_side right \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.\nQuery:" \
  --query_max_len 512 \
  --dataset_name Tevatron/beir \
  --dataset_config scifact \
  --dataset_split test \
  --encode_output_path ${output_dir}/queries_scifact.pkl \
  --encode_is_query


# Encode corpus
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --bf16 \
  --per_device_eval_batch_size 4 \
  --normalize \
  --pooling last \
  --padding_side right \
  --passage_prefix "" \
  --passage_max_len 512 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config scifact \
  --dataset_split train \
  --encode_output_path ${output_dir}/corpus_scifact.pkl \
  --passage_chunk_size 256

python -m tevatron.retriever.driver.search \
    --query_reps ${output_dir}/queries_scifact.pkl \
    --passage_reps ${output_dir}/corpus_scifact.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_dir}/rank.scifact.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input ${output_dir}/rank.scifact.txt \
                                                       --output ${output_dir}/rank.scifact.trec \
                                                       --remove_query

python -m tevatron.retriever.driver.search \
    --query_reps ${output_dir}/queries_scifact.pkl \
    --passage_reps ${output_dir}/corpus_scifact.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_dir}/rank.scifact.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input ${output_dir}/rank.scifact.txt \
                                                       --output ${output_dir}/rank.scifact.trec \
                                                       --remove_query
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-scifact-test ${output_dir}/rank.scifact.trec

# recall_100              all     0.9767
# ndcg_cut_10             all     0.7801

