SCRIPT_DIR=$PWD
cd marco
python $SCRIPT_DIR/build_hn.py --tokenizer_name bert-base-uncased --hn_file ../train.rank.txt --qrels qrels.train.tsv \
  --queries train.query.txt --collection corpus.tsv --save_to bert/train-hn
ln -s bert/train/* bert/train-hn
cd -