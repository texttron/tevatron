import json
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from multiprocessing import Manager
from tqdm import tqdm
from pyserini.eval.evaluate_dpr_retrieval import SimpleTokenizer, has_answers


class BasicHardNegativeMiner:
    def __init__(self, results_path, corpus_dataset, depth):
        self.corpus_data = corpus_dataset
        self.depth=depth
        manager = Manager()
        self.retrieval_results = manager.dict(self._read_result(results_path))
        self.docid_to_idx = manager.dict({k: v for v, k in enumerate(self.corpus_data['docid'])})

    @staticmethod
    def _read_result(path):
        retrieval_results = {}
        with open(path) as f:
            for line in f:
                qid, pid, _ = line.rstrip().split()
                if qid not in retrieval_results:
                    retrieval_results[qid] = []
                retrieval_results[qid].append(pid)
        return retrieval_results

    def __call__(self, example):
        query_id = example['query_id']
        retrieved_docid = self.retrieval_results[query_id]
        positive_ids = [pos['docid'] for pos in example['positive_passages']]
        hard_negatives = []
        for docid in retrieved_docid[:self.depth]:
            doc_info = self.corpus_data[self.docid_to_idx[docid]]
            text = doc_info['text']
            title = doc_info['title'] if 'title' in doc_info else None
            if docid not in positive_ids:
                hn_doc = {'docid': docid, 'text': text}
                if title:
                    hn_doc['title'] = title
                hard_negatives.append(hn_doc)
        example['negative_passages'] = hard_negatives
        return example


class EMHardNegativeMiner(BasicHardNegativeMiner):
    def __init__(self, results_path, corpus_dataset, depth, tokenzier, regex=False):
        self.tokenizer = tokenzier
        self.regex = regex
        super().__init__(results_path, corpus_dataset, depth)

    def __call__(self, example):
        query_id = example['query_id']
        retrieved_docid = self.retrieval_results[query_id]
        answers = example['answers']
        positives = []
        hard_negatives = []
        for docid in retrieved_docid[:self.depth]:
            doc_info = self.corpus_data[self.docid_to_idx[docid]]
            text = doc_info['text']
            title = doc_info['title'] if 'title' in doc_info else None
            if not has_answers(text, answers, self.tokenizer, self.regex):
                hn_doc = {'docid': docid, 'text': text}
                if title:
                    hn_doc['title'] = title
                hard_negatives.append(hn_doc)
            else:
                pos_doc = {'docid': docid, 'text': text}
                if title:
                    pos_doc['title'] = title
                positives.append(pos_doc)
        example['negative_passages'] = hard_negatives
        example['positive_passages'] = positives
        return example


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_data_name', type=str, required=True)
    parser.add_argument('--corpus_data_name', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--depth', type=int, default=100, required=False)
    parser.add_argument('--min_hn', type=int, default=1, required=False)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=False)
    parser.add_argument('--proc_num', type=int, default=12, required=False)
    parser.add_argument('--em', action='store_true', required=False)
    parser.add_argument('--regex', action='store_true', required=False)

    args = parser.parse_args()
    train_data = load_dataset(args.train_data_name, cache_dir=args.cache_dir)['train']
    corpus_data = load_dataset(args.corpus_data_name, cache_dir=args.cache_dir)['train']
    if args.em:
        miner = EMHardNegativeMiner(args.result_path, corpus_data, args.depth, SimpleTokenizer(), regex=args.regex)
    else:
        miner = BasicHardNegativeMiner(args.result_path, corpus_data, args.depth)

    hn_data = train_data.map(
        miner,
        batched=False,
        num_proc=args.proc_num,
        desc="Running hard negative mining",
    )

    combined_data = concatenate_datasets([train_data, hn_data])
    combined_data = combined_data.filter(
        function=lambda data: len(data["positive_passages"]) >= 1 and len(data["negative_passages"]) >= args.min_hn
    )

    with open(args.output, 'w') as f:
        for e in tqdm(combined_data):
            f.write(json.dumps(e, ensure_ascii=False)+'\n')
