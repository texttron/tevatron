from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from tevatron.preprocessor import MarcoPassageTrainPreProcessor as TrainPreProcessor

random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--negative_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()


qrel = TrainPreProcessor.read_qrel(args.qrels)


def read_line(l):
    q, nn = l.strip().split('\t')
    nn = nn.split(',')
    random.shuffle(nn)
    return q, qrel[q], nn[:args.n_sample]


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    max_length=args.truncate,
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

with open(args.negative_file) as nf:
    pbar = tqdm(map(read_line, nf))
    with Pool() as p:
        for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:02d}.json'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
                counter = 0

if f is not None:
    f.close()