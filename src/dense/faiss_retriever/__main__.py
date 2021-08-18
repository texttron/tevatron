import torch
import numpy as np
from argparse import ArgumentParser
from .retriever import BaseFaissIPRetriever


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)

    args = parser.parse_args()

    q_reps, _ = torch.load(args.query_reps)
    p_reps, p_lookup = torch.load(args.passage_reps)
    q_reps = q_reps.float().numpy()
    p_reps = p_reps.float().numpy()

    retriever = BaseFaissIPRetriever(p_reps)
    retriever.add(p_reps)
    all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)

    psg_indices = [[int(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    torch.save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()