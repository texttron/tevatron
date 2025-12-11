import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
import faiss

from tevatron.retriever.searcher import FaissFlatSearcher

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


def search_queries_chunked(retriever, q_reps, p_lookup, args):
    """
    Search with chunked passages and aggregate by document using MaxSim.
    """
    # Search more chunks to ensure good recall after aggregation
    search_depth = args.depth * args.chunk_multiplier
    
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, search_depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, search_depth)
    
    # Aggregate by document ID using MaxSim
    aggregated_results = []
    for q_idx in range(len(q_reps)):
        scores = all_scores[q_idx]
        indices = all_indices[q_idx]
        
        doc_max_scores = defaultdict(lambda: float('-inf'))
        
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for insufficient results
                continue
            
            doc_id, chunk_idx = p_lookup[idx]
            # MaxSim: keep the maximum score for each document
            doc_max_scores[doc_id] = max(doc_max_scores[doc_id], score)
        
        # Sort by score and take top-depth
        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)[:args.depth]
        aggregated_results.append(sorted_docs)
    
    return aggregated_results


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def write_ranking_chunked(results, q_lookup, ranking_save_file):
    """
    Write ranking results from chunked search.
    results: List[List[Tuple[doc_id, score]]]
    """
    with open(ranking_save_file, 'w') as f:
        for qid, doc_scores in zip(q_lookup, results):
            for doc_id, score in doc_scores:
                f.write(f'{qid}\t{doc_id}\t{score}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
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
    # Chunked search arguments
    parser.add_argument('--chunked', action='store_true',
                        help='Enable chunked search with document-level MaxSim aggregation')
    parser.add_argument('--chunk_multiplier', type=int, default=10,
                        help='Multiply search depth by this factor for chunked search to ensure recall')

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    # Auto-detect chunked format: lookup entries are tuples (doc_id, chunk_idx)
    is_chunked = args.chunked or (len(look_up) > 0 and isinstance(look_up[0], tuple))
    
    if is_chunked:
        unique_docs = len(set(doc_id for doc_id, _ in look_up))
        logger.info(f"Chunked mode: {len(look_up)} chunks from {unique_docs} documents")
        logger.info(f"Search depth: {args.depth} docs, chunk search depth: {args.depth * args.chunk_multiplier}")

    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps

    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co,
                                                     ngpu=num_gpus)

    logger.info('Index Search Start')
    
    if is_chunked:
        # Chunked search with MaxSim aggregation
        aggregated_results = search_queries_chunked(retriever, q_reps, look_up, args)
        logger.info('Index Search Finished (chunked mode with MaxSim aggregation)')
        
        if args.save_text:
            write_ranking_chunked(aggregated_results, q_lookup, args.save_ranking_to)
        else:
            # Convert to arrays for pickle
            all_scores = []
            all_doc_ids = []
            for doc_scores in aggregated_results:
                scores = [s for _, s in doc_scores]
                doc_ids = [d for d, _ in doc_scores]
                all_scores.append(scores)
                all_doc_ids.append(doc_ids)
            pickle_save((all_scores, all_doc_ids), args.save_ranking_to)
    else:
        # Standard search
        all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
        logger.info('Index Search Finished')

        if args.save_text:
            write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((all_scores, psg_indices), args.save_ranking_to) 

if __name__ == '__main__':
    main()
