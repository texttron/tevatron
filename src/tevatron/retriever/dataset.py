import random
from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from PIL import Image
from tevatron.retriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)

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
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)
    
    def _get_info_from_docid(self, docid, prefix):
        document_info = self.corpus[int(docid)]
        assert int(document_info['docid']) == int(docid)
        image = None if 'image' not in document_info else document_info['image']
        image = image.convert('RGB')
        text = None if 'text' not in document_info else document_info['text']
        text = '' if text is None else text
        return prefix + text, image

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)
        query_id = group['query_id']
        query_text = group['query_text']
        query_text = '' if query_text is None else query_text
        query_image = group['query_image']
        positive_document_ids = group['positive_document_ids']
        negative_document_ids = group['negative_document_ids']

        formated_query = (self.data_args.query_prefix + query_text, query_image)
        formated_documents = []

        selected_positive_document_id = positive_document_ids[(_hashed_seed + epoch) % len(positive_document_ids)]
        
        formated_documents.append(self._get_info_from_docid(selected_positive_document_id, self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(negative_document_ids) < negative_size:
            selected_negative_document_ids = random.choices(negative_document_ids, k=negative_size)
        elif self.data_args.train_group_size == 1:
            selected_negative_document_ids = []
        else:
            _offset = epoch * negative_size % len(negative_document_ids)
            selected_negative_document_ids = [x for x in negative_document_ids]
            random.Random(_hashed_seed).shuffle(selected_negative_document_ids)
            selected_negative_document_ids = selected_negative_document_ids * 2
            selected_negative_document_ids = selected_negative_document_ids[_offset: _offset + negative_size]

        for negative_document_id in selected_negative_document_ids:
            formated_documents.append(self._get_info_from_docid(negative_document_id, self.data_args.passage_prefix))

        return formated_query, formated_documents


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
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = format_query(text['query'], self.data_args.query_prefix)
        else:
            text_id = text['docid']
            formated_text = format_passage(text['text'], text['title'], self.data_args.passage_prefix)
        return text_id, formated_text
