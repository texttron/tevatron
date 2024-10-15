import random
from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image

from arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.corpus = load_dataset(
            self.data_args.corpus_name,
            self.data_args.corpus_config,
            data_files=self.data_args.corpus_path,
            split=self.data_args.corpus_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.docid2idx = {}
        if 'docid' in self.corpus.features:
            for idx, docid in enumerate(self.corpus['docid']):
                self.docid2idx[str(docid)] = idx
        else:
            # handle docmatix
            for idx in range(len(self.corpus)):
                self.docid2idx[str(idx)] = idx
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, docid):
        if 'image' in self.corpus.features:
            image = self.corpus[self.docid2idx[docid]]['image']
        elif 'images' in self.corpus.features:
            # handle docmatrix
            example_id, image_id = docid.split('_')
            image = self.corpus[self.docid2idx[example_id]]['images'][int(image_id)]
        return image
        

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(self._get_image(pos_psg['docid']))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(self._get_image(neg_psg['docid']))
        return formated_query, formated_passages


class EncodeDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.corpus = load_dataset(
            self.data_args.corpus_name,
            self.data_args.corpus_config,
            data_files=self.data_args.corpus_path,
            split=self.data_args.corpus_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.docid2idx = {}
        if 'docid' in self.corpus.features:
            for idx, docid in enumerate(self.corpus['docid']):
                self.docid2idx[str(docid)] = idx
        else:
            # handle docmatix
            for idx in range(len(self.corpus)):
                self.docid2idx[str(idx)] = idx
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def _get_image(self, docid):
        if 'image' in self.corpus.features:
            image = self.corpus[self.docid2idx[docid]]['image']
        elif 'images' in self.corpus.features:
            example_id, image_id = docid.split('_')
            image = self.corpus[self.docid2idx[example_id]]['images'][int(image_id)]
        return image

    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_query = format_query(text['query'], self.data_args.query_prefix)
            return text_id, formated_query
        else:
            text_id = text['docid']
            formated_passage = self._get_image(text['docid'])
            return text_id, formated_passage
