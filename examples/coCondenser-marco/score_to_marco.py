import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('score_file')
args = parser.parse_args()

with open(args.score_file) as f:
    lines = f.readlines()

all_scores = defaultdict(list)

for line in lines:
    if len(line.strip()) == 0:
        continue
    qid, did, score = line.strip().split()
    score = float(score)
    all_scores[qid].append((did, score))

qq = list(all_scores.keys())

with open(args.score_file + '.marco', 'w') as f:
    for qid in qq:
        score_list = sorted(all_scores[qid], key=lambda x: x[1], reverse=True)
        for rank, (did, score) in enumerate(score_list):
            f.write(f'{qid}\t{did}\t{rank+1}\n')

