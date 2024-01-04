import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from tevatron.reranker.arguments import DataArguments


logger = logging.getLogger(__name__)


@dataclass
class RerankerTrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[List[str]]):
        """
        Collate function for training.
        :param features: list of pairs
        :return: tokenized pairs
        """
        all_pairs = []
        for pairs in features:
            all_pairs.extend(pairs)
        
        tokenized_pairs = self.tokenizer(
            all_pairs,
            padding=False, 
            truncation=True,
            max_length=self.data_args.rerank_max_len-1 if self.data_args.append_eos_token else self.data_args.rerank_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            tokenized_pairs['input_ids'] = [p + [self.tokenizer.eos_token_id] for p in tokenized_pairs['input_ids']]
        
        pairs_collated = self.tokenizer.pad(
            tokenized_pairs,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return pairs_collated


@dataclass
class RerankerInferenceCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (query_id, text_id, pair) tuples
        """
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        pairs = [x[2] for x in features]
        collated_pairs = self.tokenizer(
            pairs,
            padding=False, 
            truncation=True,
            max_length=self.data_args.rerank_max_len-1 if self.data_args.append_eos_token else self.data_args.rerank_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_pairs['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_pairs['input_ids']]
        collated_pairs = self.tokenizer.pad(
            collated_pairs,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return query_ids, text_ids, collated_pairs