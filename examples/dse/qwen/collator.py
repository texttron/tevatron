import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin
from qwen_vl_utils import process_vision_info
from PIL import Image

from arguments import DataArguments


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    processor: ProcessorMixin


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
        
        query_messages = []
        for query in all_queries:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height': 1, 'resized_width': 1},
                        {'type': 'text', 'text': f'Query: {query}'}
                    ]
                }
            ]
            query_messages.append(message)

        passage_messages = []
        for idx in range(len(all_passages)):
            image = all_passages[idx]
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image, 'resized_height': 748, 'resized_width': 748},
                        {'type': 'text', 'text': f'What is shown in this image?'}
                    ]
                }
            ]
            passage_messages.append(message)
        
        query_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) + '<|endoftext|>'
            for msg in query_messages
        ]

        passage_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) + '<|endoftext|>'
            for msg in passage_messages
        ]

        query_image_inputs, query_video_inputs = process_vision_info(query_messages)
        passage_image_inputs, passage_video_inputs = process_vision_info(passage_messages)

        query_inputs = self.processor(
            query_texts,
            images=query_image_inputs,
            videos=query_video_inputs,
            return_tensors="pt",
            padding="longest",
        )

        passage_inputs = self.processor(
            passage_texts,
            images=passage_image_inputs,
            videos=passage_video_inputs,
            return_tensors="pt",
            padding="longest",
        )
        return query_inputs, passage_inputs
    


@dataclass
class EncodeCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        passages = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        if self.data_args.encode_is_query:
            query_messages = []
            for text in passages:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height': 1, 'resized_width': 1},
                            {'type': 'text', 'text': f'Query: {text}'}
                        ]
                    }
                ]
                query_messages.append(message)
            query_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) + '<|endoftext|>'
                for msg in query_messages
            ]
            query_image_inputs, query_video_inputs = process_vision_info(query_messages)
            inputs = self.processor(
                query_texts,
                images=query_image_inputs,
                videos=query_video_inputs,
                return_tensors="pt",
                padding="longest",
            )
        else:
            passage_messages = []
            for idx in range(len(passages)):
                image = passages[idx]
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': image, 'resized_height': 748, 'resized_width': 748},
                            {'type': 'text', 'text': f'What is shown in this image?'}
                        ]
                    }
                ]
                passage_messages.append(message)
            passage_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) + '<|endoftext|>'
                for msg in passage_messages
            ]
            passage_image_inputs, passage_video_inputs = process_vision_info(passage_messages)
            inputs = self.processor(
                passage_texts,
                images=passage_image_inputs,
                videos=passage_video_inputs,
                return_tensors="pt",
                padding="longest",
            )
        
        return text_ids, inputs
