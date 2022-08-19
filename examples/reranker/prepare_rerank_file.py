import json
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm

def read_result(path):
    retrieval_results = {}
    with open(path) as f:
        for line in f:
            if len(line.rstrip().split()) == 3:
                qid, pid, score = line.rstrip().split()
            else:
                qid, _, pid, _, score, _ = line.rstrip().split()
            if qid not in retrieval_results:
                retrieval_results[qid] = []
            retrieval_results[qid].append((pid, float(score)))
    return retrieval_results


parser = ArgumentParser()
parser.add_argument('--query_data_name', type=str, required=True)
parser.add_argument('--query_data_split', type=str, required=False, default='dev')
parser.add_argument('--corpus_data_name', type=str, required=True)
parser.add_argument('--retrieval_results', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--depth', type=int, default=1000, required=False)
parser.add_argument('--cache_dir', type=str, required=False)

args = parser.parse_args()
query_data = load_dataset(args.query_data_name, cache_dir=args.cache_dir)[args.query_data_split]
corpus_data = load_dataset(args.corpus_data_name, cache_dir=args.cache_dir)['train']
query_id_map = {}
for e in tqdm(query_data):
    query_id_map[e['query_id']] = e['query']

corpus_id_map = {}
for e in tqdm(corpus_data):
    corpus_id_map[e['docid']] = e

retrieval_results = read_result(args.retrieval_results)

with open(args.output_path, 'w') as f:
    for qid in tqdm(retrieval_results):
        query = query_id_map[qid]
        pid_and_scores = retrieval_results[qid]
        for item in pid_and_scores[:args.depth]:
            pid, score = item
            psg_info = corpus_id_map[pid]
            psg_info['score'] = score
            psg_info['query_id'] = qid
            psg_info['query'] = query
            f.write(json.dumps(psg_info)+'\n')
