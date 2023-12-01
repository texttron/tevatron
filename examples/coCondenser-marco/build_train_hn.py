from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from tevatron.preprocessor import MarcoPassageTrainPreProcessor as TrainPreProcessor

def load_ranking(rank_file, relevance, n_sample, depth):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, p_0, score_0 = next(lines).strip().split()

        curr_q = q_0
        neg_candidates = [] if p_0 in relevance[q_0] else [(p_0, float(score_0))]

        while True:
            try:
                q, p, score = next(lines).strip().split()
                if q != curr_q:
                    neg_candidates = neg_candidates[:depth]
                    random.shuffle(neg_candidates)

                    # Extract passages and scores
                    neg_passages, neg_scores = zip(*neg_candidates[:n_sample]) if neg_candidates else ([], [])

                    yield curr_q, relevance[curr_q], list(neg_passages), list(neg_scores)
                    curr_q = q
                    neg_candidates = [] if p in relevance[q] else [(p, float(score))]
                else:
                    if p not in relevance[q]:
                        #negatives.append(p)
                        neg_candidates.append((p, float(score)))
                        
            except StopIteration:
                neg_candidates = neg_candidates[:depth]
                random.shuffle(neg_candidates)

                # Extract passages and scores
                neg_passages, neg_scores = zip(*neg_candidates[:n_sample]) if neg_candidates else ([], [])

                yield curr_q, relevance[curr_q], list(neg_passages), list(neg_scores)

                return


#random.seed(datetime.now())
random.seed()
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=63)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()

qrel = TrainPreProcessor.read_qrel(args.qrels)
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

pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth))
with Pool() as p:
    for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.json'), 'w')
            pbar.set_description(f'split - {shard_id:02d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()