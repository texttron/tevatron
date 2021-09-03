SCRIPT_DIR=$PWD/../msmarco-passage-ranking/

wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
cd marco

wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
gunzip qidpidtriples.train.full.2.tsv.gz
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv
awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train.negatives.tsv

TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert

python $SCRIPT_DIR/build_train.py --tokenizer_name $TOKENIZER --negative_file train.negatives.tsv --qrels qrels.train.tsv \
  --queries train.query.txt --collection corpus.tsv --save_to ${TOKENIZER_ID}/train
python $SCRIPT_DIR/tokenize_queries.py --tokenizer_name $TOKENIZER --query_file dev.query.txt --save_to $TOKENIZER_ID/query/dev.query.json
python $SCRIPT_DIR/tokenize_queries.py --tokenizer_name $TOKENIZER --query_file train.query.txt --save_to $TOKENIZER_ID/query/train.query.json
python $SCRIPT_DIR/tokenize_passages.py --tokenizer_name $TOKENIZER --file corpus.tsv --save_to $TOKENIZER_ID/corpus

cd -