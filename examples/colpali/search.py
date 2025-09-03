import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
from transformers import ColPaliProcessor
import torch

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file, depth):
    with open(ranking_save_file, "w") as f:
        for qid, q_doc_scores, q_doc_indices in zip(
            q_lookup, corpus_scores, corpus_indices
        ):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list[:depth]:
                f.write(f"{qid}\t{idx}\t{s}\n")


def pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return reps, lookup


def pickle_save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument("--query_reps", required=True)
    parser.add_argument("--passage_reps", required=True)
    parser.add_argument("--processor_name", default="vidore/colpali-v1.2-hf")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--save_ranking_to", required=True)
    parser.add_argument("--save_text", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps.to("cuda")

    retriever = ColPaliProcessor.from_pretrained(args.processor_name)

    index_files = glob.glob(args.passage_reps)
    logger.info(f"Pattern match found {len(index_files)} shards.")
    p_reps_0, p_lookup_0 = pickle_load(index_files[0])

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc="Scoring shards", total=len(index_files))
    psg_indices = []
    all_scores = []
    for reps, look_up in shards:
        reps = reps.to("cuda")
        scores = retriever.score_retrieval(q_reps, reps)  # (num_queries, num_passages)
        # torch sort by scores
        scores, indices = torch.sort(scores, dim=1, descending=True)
        # depth cutoff
        scores = scores[:, : args.depth]
        indices = indices[:, : args.depth]
        # get docid by indices (2D)
        indices = indices.cpu().numpy()
        docids = np.vectorize(lambda x: look_up[x])(indices)
        all_scores.append(scores.cpu().numpy())
        psg_indices.append(docids)

    all_scores = np.concatenate(all_scores, axis=1)
    psg_indices = np.concatenate(psg_indices, axis=1)
    if args.save_text:
        write_ranking(
            psg_indices, all_scores, q_lookup, args.save_ranking_to, args.depth
        )
    else:
        pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == "__main__":
    main()
