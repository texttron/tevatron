import torch
import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm

from .retriever import BaseFaissIPRetriever
from .reducer import write_ranking

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[int(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = torch.load(index_files[0])
    retriever = BaseFaissIPRetriever(p_reps_0.float().numpy())

    shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps.float().numpy())
        look_up += p_lookup

    q_reps, q_lookup = torch.load(args.query_reps)
    q_reps = q_reps.float().numpy()

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        torch.save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()