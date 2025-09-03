import random
import os
from typing import List, Tuple

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from PIL import Image

from tevatron.retriever.arguments import DataArguments

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    """
    Dataset for training which handles both query and passage data.
    Loads dataset and optional corpus from the provided paths/configurations.
    """

    def __init__(
        self,
        data_args: DataArguments,
        trainer=None,
        dataset_name=None,
        corpus_name=None,
        dataset_path=None,
        corpus_path=None,
        corpus_assets_path=None,
    ):
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=(
                self.data_args.dataset_path if dataset_path is None else dataset_path
            ),
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )

        # Load corpus if provided
        if self.data_args.corpus_name is None and corpus_name is None:
            self.corpus = None
        else:
            self.corpus = load_dataset(
                self.data_args.corpus_name if corpus_name is None else corpus_name,
                self.data_args.corpus_config,
                data_files=(
                    self.data_args.corpus_path if corpus_path is None else corpus_path
                ),
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )

        # for video we use assets_path to load the video
        self.corpus_assets_path = (
            corpus_assets_path
            if corpus_assets_path is not None
            else self.data_args.assets_path
        )

        # create a map between docid and index
        self.docid_to_index = {}
        if self.corpus is not None:
            corpus_ids = self.corpus.select_columns(["docid"])
            docids = corpus_ids["docid"]
            self.docid_to_index = {
                docid: index for index, docid in enumerate(tqdm(docids))
            }

    def set_trainer(self, trainer):
        """Sets the trainer for the dataset."""
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_info_from_docid(self, docid, prefix):
        """
        Retrieves document information from the corpus given a docid.
        Returns:
            tuple: (formatted_text, image, video, audio)
        """
        document_info = self.corpus[self.docid_to_index[docid]]
        assert document_info["docid"] == docid
        image = document_info.get("image", None)

        video = document_info.get("video", None)
        if video is not None:
            video = os.path.join(self.corpus_assets_path, video)

        audio = document_info.get("audio", None)
        if audio is not None:  # either an dict with 'array' key or a string .mp3 path
            if isinstance(audio, dict) and "array" in audio:
                audio = audio["array"]
            else:
                assert isinstance(audio, str) and audio.endswith(".mp3")
                audio = os.path.join(self.corpus_assets_path, audio)

        text = document_info.get("text", "")

        if not self.data_args.encode_text:
            text = None
        if not self.data_args.encode_image:
            image = None
        if not self.data_args.encode_video:
            video = None
        if not self.data_args.encode_audio:
            audio = None
        text = "" if text is None else text

        return prefix + text, image, video, audio

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        # Handling the legacy format with 'positive_passages'
        if "positive_passages" in group:
            query_text = group["query"]
            query_image = query_video = query_audio = None
            formatted_query = (
                self.data_args.query_prefix + query_text,
                query_image,
                query_video,
                query_audio,
            )

            formatted_documents = []
            # Select positive document
            selected_positive = group["positive_passages"][
                (_hashed_seed + epoch) % len(group["positive_passages"])
            ]
            positive_text = (
                selected_positive["title"] + " " + selected_positive["text"]
                if "title" in selected_positive
                else selected_positive["text"]
            )
            formatted_documents.append(
                (self.data_args.passage_prefix + positive_text, None, None, None)
            )

            # Select negative documents
            negative_size = self.data_args.train_group_size - 1
            if len(group["negative_passages"]) < negative_size:
                selected_negatives = random.choices(
                    group["negative_passages"], k=negative_size
                )
            elif self.data_args.train_group_size == 1:
                selected_negatives = []
            else:
                offset = epoch * negative_size % len(group["negative_passages"])
                selected_negatives = list(group["negative_passages"])
                random.Random(_hashed_seed).shuffle(selected_negatives)
                selected_negatives = selected_negatives * 2
                selected_negatives = selected_negatives[offset : offset + negative_size]

            for negative in selected_negatives:
                negative_text = (
                    negative["title"] + " " + negative["text"]
                    if "title" in negative
                    else negative["text"]
                )
                formatted_documents.append(
                    (self.data_args.passage_prefix + negative_text, None, None, None)
                )

            return formatted_query, formatted_documents

        # Handling the new format
        query_id = group["query_id"]
        query_text = group.get("query_text", "") or ""
        query_image = group.get("query_image", None)
        query_video = group.get("query_video", None)
        query_audio = group.get("query_audio", None)
        formatted_query = (
            self.data_args.query_prefix + query_text,
            query_image,
            query_video,
            query_audio,
        )

        formatted_documents = []
        positive_document_ids = group["positive_document_ids"]
        negative_document_ids = group["negative_document_ids"]

        # Select positive document id
        selected_positive_docid = positive_document_ids[
            (_hashed_seed + epoch) % len(positive_document_ids)
        ]
        formatted_documents.append(
            self._get_info_from_docid(
                selected_positive_docid, self.data_args.passage_prefix
            )
        )

        # Select negative document ids
        negative_size = self.data_args.train_group_size - 1
        if len(negative_document_ids) < negative_size:
            selected_negative_docids = random.choices(
                negative_document_ids, k=negative_size
            )
        elif self.data_args.train_group_size == 1:
            selected_negative_docids = []
        else:
            offset = epoch * negative_size % len(negative_document_ids)
            selected_negative_docids = list(negative_document_ids)
            random.Random(_hashed_seed).shuffle(selected_negative_docids)
            selected_negative_docids = selected_negative_docids * 2
            selected_negative_docids = selected_negative_docids[
                offset : offset + negative_size
            ]

        for neg_docid in selected_negative_docids:
            formatted_documents.append(
                self._get_info_from_docid(neg_docid, self.data_args.passage_prefix)
            )

        return formatted_query, formatted_documents


class MultiTrainDataset(Dataset):
    """
    Dataset for training from multiple datasets.
    Iterates over a list of datasets and their corresponding corpora.
    """

    def __init__(
        self,
        data_args: DataArguments,
        dataset_list=None,
        corpus_list=None,
        trainer=None,
    ):
        self.data_args = data_args
        self.trainer = trainer
        self.train_datasets = []

        for ds_entry, corpus_entry in zip(dataset_list, corpus_list):
            ds_path = ds_entry["name"]
            corpus_path = corpus_entry["name"]
            corpus_assets_path = corpus_entry["assets_path"]
            dataset_name = None
            corpus_name = None
            ds_file = None
            corpus_file = None

            # Determine dataset type
            if os.path.isdir(ds_path):
                dataset_name = ds_path
            elif ds_path.endswith(".jsonl"):
                dataset_name = "json"
                ds_file = ds_path
            else:
                dataset_name = ds_path

            # Determine corpus type
            if corpus_path is None:
                corpus_name = None
            elif os.path.isdir(corpus_path):
                corpus_name = corpus_path
            elif corpus_path.endswith(".jsonl"):
                corpus_name = "json"
                corpus_file = corpus_path
            else:
                corpus_name = corpus_path

            self.train_datasets.append(
                TrainDataset(
                    self.data_args,
                    self.trainer,
                    dataset_name,
                    corpus_name,
                    dataset_path=ds_file,
                    corpus_path=corpus_file,
                    corpus_assets_path=corpus_assets_path,
                )
            )

    def __len__(self):
        return sum(len(dataset) for dataset in self.train_datasets)

    def __getitem__(self, item):
        dataset_index = 0
        while item >= len(self.train_datasets[dataset_index]):
            item -= len(self.train_datasets[dataset_index])
            dataset_index += 1
        return self.train_datasets[dataset_index][item]

    def set_trainer(self, trainer):
        """Sets the trainer for all sub-datasets."""
        self.trainer = trainer
        for dataset in self.train_datasets:
            dataset.set_trainer(trainer)


class EncodeDataset(Dataset):
    """
    Dataset for encoding.
    Loads data and optionally shards it for distributed processing.
    """

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        content = self.encode_data[item]
        if self.data_args.encode_is_query:
            content_id = content["query_id"]
            content_text = content.get("query_text", content.get("query", ""))
            content_text = self.data_args.query_prefix + content_text
            content_image = content.get("query_image", None)
            content_video = content.get("query_video", None)
            content_audio = content.get("query_audio", None)
        else:
            content_id = content["docid"]
            content_text = content.get("text", "")
            if "title" in content:
                content_text = content["title"] + " " + content_text
            content_text = self.data_args.passage_prefix + content_text.strip()
            content_image = content.get("image", None)
            content_video = content.get("video", None)
            content_audio = content.get("audio", None)

        if content_video is not None and self.data_args.encode_video:
            content_video = os.path.join(self.data_args.assets_path, content_video)
            # check if the file exists
            if not os.path.exists(content_video):
                logger.warning(f"Video file {content_video} does not exist.")
                content_video = None

        if (
            content_audio is not None
        ):  # either an dict with 'array' key or a string .mp3 path
            if isinstance(content_audio, dict) and "array" in content_audio:
                content_audio = content_audio["array"]
            else:
                assert isinstance(content_audio, str) and content_audio.endswith(".mp3")
                content_audio = os.path.join(self.data_args.assets_path, content_audio)
                # check if the file exists
                if not os.path.exists(content_audio):
                    logger.warning(f"Audio file {content_audio} does not exist.")
                    content_audio = None

        if not self.data_args.encode_text:
            content_text = None
        if not self.data_args.encode_image:
            content_image = None
        if not self.data_args.encode_video:
            content_video = None
        if not self.data_args.encode_audio:
            content_audio = None

        return content_id, content_text, content_image, content_video, content_audio
