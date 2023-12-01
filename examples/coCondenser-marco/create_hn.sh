SCRIPT_DIR=$PWD
cd marco
python $SCRIPT_DIR/build_train_hn.py --tokenizer_name bert-base-uncased --hn_file ../train.rank.tsv --qrels qrels.train.tsv --queries train.query.txt --collection corpus.tsv --save_to /data/cme/marco/bert/train-hn-withscore
ln -s /data/cme/marco/bert/train/* /data/cme/marco/bert/train-hn-withscore
cd -