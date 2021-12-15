import argparse
import os

parser = argparse.ArgumentParser(description='Reduce retrieval results from multiple shards.')
parser.add_argument('--results_dir', type=str, help='Directory that contains results from all shards', required=True)
parser.add_argument('--output', help='Path to final results file', required=True)
parser.add_argument('--depth', type=int, help='Number of retrieved doc for each query', required=False, default=100)
args = parser.parse_args()


all_results = {}
print(f'Merging results from {len(os.listdir(args.results_dir))} result files.')
for filename in os.listdir(args.results_dir):
    path = os.path.join(args.results_dir, filename)
    with open(path) as f:
        for line in f:
            qid, docid, score = line.split()
            score = float(score)
            if qid not in all_results:
                all_results[qid] = []
            all_results[qid].append((docid, score))

with open(args.output, 'w') as f:
    print(f'Writing output to {args.output} with depth {args.depth}')
    for qid in all_results:
        results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)[:args.depth]
        for docid, score in results:
            f.write(f'{qid}\t{docid}\t{score}\n')
