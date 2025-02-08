import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from qwen_vl_utils import process_vision_info
from PIL import Image

from tevatron.retriever.arguments import DataArguments


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    """
    simple collator for text only data.
    """
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        all_queries = [q[0] for q in all_queries]
        all_passages = [p[0] for p in all_passages]
        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=False, 
            truncation=True,
            max_length=self.data_args.passage_max_len-1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]
        
        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return q_collated, d_collated


@dataclass
class MultiModalTrainCollator:
    """
    collator for text-visual data.
    """
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: prepared model inputs
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        
        query_messages = []
        for query in all_queries:
            text = query[0]
            image = query[1]
            content = []
            if text:
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(text, max_length=self.data_args.query_max_len, truncation=True)
                )
                content.append({'type': 'text', 'text': text})
            if image:
                content.append({'type': 'image', 'image': image, 'resized_height': 784, 'resized_width': 784})
            message = [
                {
                    'role': 'user',
                    'content': content
                }
            ]
            query_messages.append(message)

        passage_messages = []
        for idx in range(len(all_passages)):
            text = all_passages[idx][0]
            image = all_passages[idx][1]
            content = []
            if text:
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(text, max_length=self.data_args.passage_max_len, truncation=True)
                )
                content.append({'type': 'text', 'text': text})
            if image:
                content.append({'type': 'image', 'image': image, 'resized_height': 784, 'resized_width': 784})
            message = [
                {
                    'role': 'user',
                    'content': content
                }
            ]
            passage_messages.append(message)
        
        query_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in query_messages
        ]

        passage_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in passage_messages
        ]
        
        if self.data_args.append_eos_token:
            # should already have a eos token so not very necessary to have additional one.
            query_texts = [x + '<|endoftext|>' for x in query_texts]
            passage_texts = [x + '<|endoftext|>' for x in passage_texts]
        

        query_image_inputs, query_video_inputs = process_vision_info(query_messages)
        passage_image_inputs, passage_video_inputs = process_vision_info(passage_messages)

        query_inputs = self.processor(
            text=query_texts,
            images=query_image_inputs,
            videos=query_video_inputs,
            return_tensors="pt",
            padding="longest",
        )

        passage_inputs = self.processor(
            text=passage_texts,
            images=passage_image_inputs,
            videos=passage_video_inputs,
            return_tensors="pt",
            padding="longest",
        )
        return query_inputs, passage_inputs

@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=False, 
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_texts['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_texts['input_ids']]
        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts


@dataclass
class VllmEncodeCollator(EncodeCollator):
    def __call__(self, features: List[Tuple[str, str]]):
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_texts['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_texts['input_ids']]
        return text_ids, collated_texts['input_ids']