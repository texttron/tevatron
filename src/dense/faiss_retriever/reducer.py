import glob
import torch
import faiss
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Iterable, Tuple
from numpy import ndarray


def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]]):
    rh = None
    for scores, indices in results:
        if rh is None:
            print(f'Initializing Heap. Assuming {scores.shape[0]} queries.')
            rh = faiss.ResultHeap(scores.shape[0], scores.shape[1])
        rh.add_result(-scores, indices)
    rh.finalize()
    corpus_scores, corpus_indices = -rh.D, rh.I

    return corpus_scores, corpus_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--score_dir', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--save_ranking_to', required=True)
    args = parser.parse_args()

    partitions = glob.glob(f'{args.score_dir}/*')

    corpus_scores, corpus_indices = combine_faiss_results(map(torch.load, tqdm(partitions)))

    _, q_lookup = torch.load(args.query)
    write_ranking(corpus_indices, corpus_scores, q_lookup, args.save_ranking_to)


if __name__ == '__main__':
    main()