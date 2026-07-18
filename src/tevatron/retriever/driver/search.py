import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import faiss

from tevatron.retriever.searcher import FaissFlatSearcher
from tevatron.utils.io import ensure_parent_dir

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def merge_results(current_scores, current_indices, shard_scores, shard_indices, depth):
    if current_scores is None:
        order = np.argsort(-shard_scores, axis=1)[:, :depth]
        return np.take_along_axis(shard_scores, order, axis=1), np.take_along_axis(shard_indices, order, axis=1)

    scores = np.concatenate([current_scores, shard_scores], axis=1)
    indices = np.concatenate([current_indices, shard_indices], axis=1)
    order = np.argsort(-scores, axis=1)[:, :depth]
    return np.take_along_axis(scores, order, axis=1), np.take_along_axis(indices, order, axis=1)


def search_queries_by_shard(index_files, q_reps, args):
    all_scores = None
    all_indices = None
    all_lookup = []
    offset = 0
    num_gpus = 0 if args.cpu_search else faiss.get_num_gpus()

    shard_iter = tqdm(index_files, desc='Searching shards', disable=args.quiet)
    for index_file in shard_iter:
        p_reps, p_lookup = pickle_load(index_file)
        retriever = FaissFlatSearcher(p_reps)
        retriever.add(p_reps)

        if num_gpus == 0:
            logger.info("No GPU found or using faiss-cpu. Searching shard on CPU.")
        else:
            logger.info(f"Using {num_gpus} GPU for shard {index_file}")
            co = faiss.GpuClonerOptions()
            co.useFloat16 = not args.gpu_float32
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)

        if args.batch_size > 0:
            shard_scores, shard_local_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
        else:
            shard_scores, shard_local_indices = retriever.search(q_reps, args.depth)
        shard_indices = np.arange(offset, offset + len(p_lookup), dtype=np.int64)[shard_local_indices]
        all_scores, all_indices = merge_results(all_scores, all_indices, shard_scores, shard_indices, args.depth)
        all_lookup += p_lookup
        offset += len(p_lookup)

    psg_indices = np.array([[str(all_lookup[x]) for x in q_dd] for q_dd in all_indices])
    return all_scores, psg_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    ensure_parent_dir(ranking_save_file)
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    ensure_parent_dir(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--shard_search', action='store_true')
    parser.add_argument('--gpu_float32', action='store_true')
    parser.add_argument('--cpu_search', action='store_true')

    args = parser.parse_args()

    ensure_parent_dir(args.save_ranking_to)

    index_files = sorted(glob.glob(args.passage_reps))
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    q_reps, q_lookup = pickle_load(args.query_reps)

    if args.shard_search:
        logger.info('Shard Search Start')
        all_scores, psg_indices = search_queries_by_shard(index_files, q_reps, args)
        logger.info('Shard Search Finished')
        if args.save_text:
            write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((all_scores, psg_indices), args.save_ranking_to)
        return

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    num_gpus = 0 if args.cpu_search else faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = not args.gpu_float32
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = not args.gpu_float32
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co,
                                                     ngpu=num_gpus)

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()
