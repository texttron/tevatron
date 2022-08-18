import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from tevatron.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class RerankerTrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, query_encoding: List[int], text_encoding: List[int]):
        item = self.tok.prepare_for_model(
            query_encoding,
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len + self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        qry = group['query']
        group_positives = group['positives']
        group_negatives = group['negatives']
        encoded_pairs = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = random.sample(group_positives, 1)[0]
        encoded_pairs.append(self.create_one_example(qry, pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            negs = random.sample(group_negatives, negative_size)
        for neg_psg in negs:
            encoded_pairs.append(self.create_one_example(qry, neg_psg))

        return encoded_pairs


class RerankerEncodeDataset(Dataset):
    input_keys = ['query_id', 'query', 'text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        query_id, query, text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_pair = self.tok.prepare_for_model(
            query,
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=True,
        )
        return query_id, text_id, encoded_pair


@dataclass
class RerankerTrainCollator(DataCollatorWithPadding):

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        pp = sum(features, [])
        pair_collated = self.tokenizer.pad(
            pp,
            padding='max_length',
            max_length=self.max_q_len+self.max_p_len,
            return_tensors="pt",
        )
        return pair_collated

@dataclass
class RerankerInferenceCollator(DataCollatorWithPadding):
    def __call__(self, features):
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        text_features = [x[2] for x in features]
        collated_features = super().__call__(text_features)
        return query_ids, text_ids, collated_features
