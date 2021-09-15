import json
import csv
import datasets
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class SimpleTrainPreProcessor:
    query_file: str
    collection_file: str
    tokenizer: PreTrainedTokenizer

    max_length: int = 128
    columns = ['text_id', 'title', 'text']
    title_field = 'title'
    text_field = 'text'

    def __post_init__(self):
        self.queries = self.read_queries(self.query_file)
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter='\t',
        )['train']

    @staticmethod
    def read_queries(queries):
        qmap = {}
        with open(queries) as f:
            for l in f:
                qid, qry = l.strip().split('\t')
                qmap[qid] = qry
        return qmap

    @staticmethod
    def read_qrel(relevance_file):
        qrel = {}
        with open(relevance_file, encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
        return qrel

    def get_query(self, q):
        query_encoded = self.tokenizer.encode(
            self.queries[q],
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        return query_encoded

    def get_passage(self, p):
        entry = self.collection[int(p)]
        title = entry[self.title_field]
        title = "" if title is None else title
        body = entry[self.text_field]
        content = title + self.tokenizer.sep_token + body

        passage_encoded = self.tokenizer.encode(
            content,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )

        return passage_encoded

    def process_one(self, train):
        q, pp, nn = train
        train_example = {
            'query': self.get_query(q),
            'positives': [self.get_passage(p) for p in pp],
            'negatives': [self.get_passage(n) for n in nn],
        }

        return json.dumps(train_example)


@dataclass
class SimpleCollectionPreProcessor:
    tokenizer: PreTrainedTokenizer
    separator: str = '\t'
    max_length: int = 128

    def process_line(self, line: str):
        xx = line.strip().split(self.separator)
        text_id, text = xx[0], xx[1:]
        text_encoded = self.tokenizer.encode(
            self.tokenizer.sep_token.join(text),
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        encoded = {
            'text_id': text_id,
            'text': text_encoded
        }
        return json.dumps(encoded)
