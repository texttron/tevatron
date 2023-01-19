import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tevatron.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class DistilPreProcessor:
    def __init__(self, student_tokenizer, teacher_tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        student_query = self.student_tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        teacher_query = self.teacher_tokenizer.encode(example['query'],
                                    add_special_tokens=False,
                                    max_length=self.query_max_length,
                                    truncation=True)
        student_positives = []
        teacher_positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            student_positives.append(self.student_tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
            teacher_positives.append(self.teacher_tokenizer.encode(text,
                                                add_special_tokens=False,
                                                max_length=self.text_max_length,
                                                truncation=True))
        student_negatives = []
        teacher_negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            student_negatives.append(self.student_tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
            teacher_negatives.append(self.teacher_tokenizer.encode(text,
                                                add_special_tokens=False,
                                                max_length=self.text_max_length,
                                                truncation=True))
        return {
            'student_query': student_query,
            'student_positives': student_positives,
            'student_negatives': student_negatives,
            'teacher_query': teacher_query,
            'teacher_positives': teacher_positives,
            'teacher_negatives': teacher_negatives
            }


class HFDistilTrainDataset:
    def __init__(self, student_tokenizer: PreTrainedTokenizer, teacher_tokenizer: PreTrainedTokenizer,
                       data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = None if data_args.dataset_name == 'json' else DistilPreProcessor
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = ' '

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.student_tokenizer, self.teacher_tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class DistilTrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            student_tokenizer: PreTrainedTokenizer,
            teacher_tokenizer: PreTrainedTokenizer
    ):
        self.train_data = dataset
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_student_example(self, text_encoding: List[int], is_query=False):
        item = self.student_tokenizer.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def create_teacher_example(self, query_encoding: List[int], text_encoding: List[int]):
        item = self.teacher_tokenizer.prepare_for_model(
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

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], List[BatchEncoding]]:
        group = self.train_data[item]
        student_qry = group['student_query']
        student_positives = group['student_positives']
        student_negatives = group['student_negatives']
        teacher_qry = group['teacher_query']
        teacher_positives = group['teacher_positives']
        teacher_negatives = group['teacher_negatives']
        
        encoded_student_query = self.create_student_example(student_qry, is_query=True)
        encoded_student_passages = []
        encoded_teacher_pairs = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg_idx = 0
        else:
            pos_psg_idx = random.sample(list(range(len(student_positives))), 1)[0]
        encoded_teacher_pairs.append(self.create_teacher_example(teacher_qry, teacher_positives[pos_psg_idx]))
        encoded_student_passages.append(self.create_student_example(student_positives[pos_psg_idx]))

        negative_size = self.data_args.train_n_passages - 1
        if len(student_negatives) < negative_size:
            negs_idxs = random.choices(list(range(len(student_negatives))), k=negative_size)
        elif self.data_args.negative_passage_no_shuffle:
            negs_idxs = list(range(len(student_negatives)))[:negative_size]
        else:
            negs_idxs = random.sample(list(range(len(student_negatives))), negative_size)
        for neg_psg_idx in negs_idxs:
            encoded_teacher_pairs.append(self.create_teacher_example(teacher_qry, teacher_negatives[neg_psg_idx]))
            encoded_student_passages.append(self.create_student_example(student_negatives[neg_psg_idx]))
        return encoded_student_query, encoded_student_passages, encoded_teacher_pairs


@dataclass
class DistilTrainCollator:
    
    tokenizer: PreTrainedTokenizerBase
    teacher_tokenizer: PreTrainedTokenizerBase
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        pp = [f[2] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        if isinstance(pp[0], list):
            pp = sum(pp, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        p_collated = self.teacher_tokenizer.pad(
            pp,
            padding='max_length',
            max_length=self.max_q_len+self.max_p_len,
            return_tensors="pt",
        )
        return q_collated, d_collated, p_collated
