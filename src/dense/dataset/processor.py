import os

import datasets
from transformers import AutoTokenizer


class DataProcessor:
    data = None

    def load(self, **kwargs):
        pass

    def process(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass


class SimpleTrainProcessor(DataProcessor):
    data = None

    def load(self, name):
        self.data = datasets.load_dataset(name, "queries")['train']

    def process(self, tokenizer, with_title=False, threads=16, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        def preprocess_function(example):
            query = tokenizer.encode(example['query'], add_special_tokens=False)
            positives = []
            for pos in example['positive_passages']:
                text = pos['title'] + " " + pos['text'] if with_title else pos['text']
                positives.append(tokenizer.encode(text, add_special_tokens=False))
            negatives = []
            for neg in example['negative_passages']:
                text = neg['title'] + " " + neg['text'] if with_title else neg['text']
                negatives.append(tokenizer.encode(text, add_special_tokens=False))
            return {'query': query, 'positives': positives, 'negatives': negatives}

        self.data = self.data.map(
            preprocess_function,
            batched=False,
            num_proc=threads,
            remove_columns=self.data.column_names,
            desc="Running tokenizer on train dataset",
        )

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.data.to_json(os.path.join(directory, 'data.json'))


class SimpleDevProcessor(SimpleTrainProcessor):
    def load(self, name):
        self.data = datasets.load_dataset(name, "queries")['dev']


class SimpleTestProcessor(DataProcessor):
    data = None

    def load(self, name):
        self.data = datasets.load_dataset(name, "queries")['test']

    def process(self, tokenizer, with_title=False, threads=16, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        def preprocess_function(example):
            query_id = example['query_id']
            query = tokenizer.encode(example['query'], add_special_tokens=False)
            answers = example['answers']
            return {'text_id': query_id, 'text': query, 'answers': answers}

        self.data = self.data.map(
            preprocess_function,
            batched=False,
            num_proc=threads,
            remove_columns=self.data.column_names,
            desc="Running tokenization on test set",
        )

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.data.to_json(os.path.join(directory, 'data.json'))


class SimpleCorpusProcessor(DataProcessor):
    data = None

    def load(self, name):
        self.data = datasets.load_dataset(name, "corpus")['train']

    def process(self, tokenizer, with_title=False, threads=16, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        def preprocess_function(example):
            docid = example['docid']
            text = example['title'] + " " + example['text'] if with_title else example['text']
            text = tokenizer.encode(text, add_special_tokens=False)
            return {'text_id': docid, 'text': text}

        self.data = self.data.map(
            preprocess_function,
            batched=False,
            num_proc=threads,
            remove_columns=self.data.column_names,
            desc="Running tokenizer on corpus",
        )

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.data.to_json(os.path.join(directory, 'corpus.json'))


class MsMarcoDevProcessor(SimpleDevProcessor):

    def process(self, tokenizer, with_title=False, threads=16, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        def preprocess_function(example):
            query_id = example['query_id']
            query = tokenizer.encode(example['query'], add_special_tokens=False)
            return {'text_id': query_id, 'text': query}

        self.data = self.data.map(
            preprocess_function,
            batched=False,
            num_proc=threads,
            remove_columns=self.data.column_names,
            desc="Running tokenization on dev set",
        )
