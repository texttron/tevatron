import logging
import torch
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from qwen_omni_utils import process_mm_info
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
        :return: tokenized query_ids, passage_ids, [eos_positions if chunked]
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        all_queries = [q[0] for q in all_queries]
        all_passages = [p[0] for p in all_passages]
        
        # Query tokenization
        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Passage tokenization
        if self.data_args.passage_chunk_size > 0:
            d_collated, eos_positions = self._tokenize_chunked_passages(all_passages)
            return q_collated, d_collated, eos_positions
        else:
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
                d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]
            d_collated = self.tokenizer.pad(
                d_collated,
                padding=True, 
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return q_collated, d_collated

    def _tokenize_chunked_passages(self, passages: List[str]):
        """
        Tokenize passages with EOS separators between chunks.
        Each chunk ends with EOS, enabling extraction of chunk embeddings from EOS positions.
        """
        chunk_size = self.data_args.passage_chunk_size
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        all_input_ids = []
        all_eos_positions = []
        
        for passage in passages:
            tokens = self.tokenizer.encode(passage, add_special_tokens=False)
            
            new_tokens = []
            eos_positions = []
            for i in range(0, max(len(tokens), 1), chunk_size):
                chunk = tokens[i:i + chunk_size]
                new_tokens.extend(chunk)
                new_tokens.append(eos_id)
                eos_positions.append(len(new_tokens) - 1)
            
            all_input_ids.append(new_tokens)
            all_eos_positions.append(eos_positions)
        
        # Padding
        max_len = min(max(len(ids) for ids in all_input_ids), self.data_args.passage_max_len)
        if self.data_args.pad_to_multiple_of:
            max_len = ((max_len + self.data_args.pad_to_multiple_of - 1) 
                       // self.data_args.pad_to_multiple_of * self.data_args.pad_to_multiple_of)
        
        padded_ids, padded_mask, final_eos_positions = [], [], []
        for input_ids, eos_pos in zip(all_input_ids, all_eos_positions):
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                eos_pos = [p for p in eos_pos if p < max_len]
            pad_len = max_len - len(input_ids)
            padded_ids.append(input_ids + [pad_id] * pad_len)
            padded_mask.append([1] * len(input_ids) + [0] * pad_len)
            final_eos_positions.append(eos_pos)
        
        d_collated = {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_mask, dtype=torch.long),
        }
        return d_collated, final_eos_positions


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
            video = query[2]
            audio = query[3]
            content = []
            if text:
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(text, max_length=self.data_args.query_max_len, truncation=True)
                )
                content.append({'type': 'text', 'text': text})
            if image:
                content.append({'type': 'image', 'image': image, 'resized_height': 784, 'resized_width': 784})
            if video:
                content.append({'type': 'video', 'video': video, 'nframes': 24, "resized_height": 280,
                                "resized_width": 280})
            if audio is not None:
                content.append({'type': 'audio', 'audio': audio, "resized_height": 280, "resized_width": 280})
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
            video = all_passages[idx][2]
            audio = all_passages[idx][3]
            content = []
            if text:
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(text, max_length=self.data_args.passage_max_len, truncation=True)
                )
                content.append({'type': 'text', 'text': text})
            if image:
                content.append({'type': 'image', 'image': image, 'resized_height': 784, 'resized_width': 784})
            if video:
                content.append({'type': 'video', 'video': video, 'nframes': 24, "resized_height": 280,
                                "resized_width": 280})
            if audio is not None:
                content.append({'type': 'audio', 'audio': audio})
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
            query_texts = [x[0] + '<|endoftext|>' for x in query_texts]
            passage_texts = [x[0] + '<|endoftext|>' for x in passage_texts]
        

        # audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        query_audio_inputs, query_image_inputs, query_video_inputs = process_mm_info(query_messages, use_audio_in_video=False)

        passage_audio_inputs, passage_image_inputs, passage_video_inputs = process_mm_info(passage_messages, use_audio_in_video=False)

        query_inputs = self.processor(
            text=query_texts,
            audio=query_audio_inputs,
            images=query_image_inputs,
            videos=query_video_inputs,
            return_tensors="pt",
            padding="longest",
        )

        passage_inputs = self.processor(
            text=passage_texts,
            audio=passage_audio_inputs,
            images=passage_image_inputs,
            videos=passage_video_inputs,
            return_tensors="pt",
            padding="longest",
        )
        return query_inputs, passage_inputs

@dataclass
class EncodeCollator:
    """
    simple collator for text only data.
    """
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        content_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[2] for x in features] # this will be ignored
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_inputs = self.tokenizer(
            texts,
            padding=False, 
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_inputs['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_inputs['input_ids']]
        collated_inputs = self.tokenizer.pad(
            collated_inputs,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return content_ids, collated_inputs

@dataclass
class MultiModalEncodeCollator:
    """
    collator for text-visual data.
    """
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        content_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[2] for x in features]
        videos = [x[3] for x in features]
        audios = [x[4] for x in features]
        messages = []
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        for idx in range(len(texts)):
            text = texts[idx]
            image = images[idx]
            video = videos[idx]
            audio = audios[idx]
            content = []
            if text:
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(text, max_length=max_length, truncation=True)
                )
                content.append({'type': 'text', 'text': text})
            if image:
                content.append({'type': 'image', 'image': image, 'resized_height': 784, 'resized_width': 784})
            if video:
                content.append({'type': 'video', 'video': video, 'nframes': 24, "resized_height": 280,
                                "resized_width": 280})
            if audio is not None:
                content.append({'type': 'audio', 'audio': audio})
            message = [
                {
                    'role': 'user',
                    'content': content
                }
            ]
            messages.append(message)
        
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        
        if self.data_args.append_eos_token:
            texts = [x[0] + '<|endoftext|>' for x in texts]

        audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=False)

        collated_inputs = self.processor(
            text=texts,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="longest",
        )
        return content_ids, collated_inputs


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


@dataclass
class VllmMultiModalEncodeCollator(MultiModalEncodeCollator):
    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        content_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[2] for x in features]
        messages = []
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        for idx in range(len(texts)):
            text = texts[idx]
            image = images[idx]
            content = []
            if text:
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(text, max_length=max_length, truncation=True)
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
            messages.append(message)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]

        if self.data_args.append_eos_token:
            texts = [x[0] + '<|endoftext|>' for x in texts]


        audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=False)
        
        return content_ids, texts, image_inputs


@dataclass
class DistilTrainCollator:
    """
    collator for distillation data.
    """
    tokenizer: PreTrainedTokenizer
    data_args: DataArguments
    torch_dtype: torch.dtype = torch.bfloat16

    def __call__(self, features: List[Tuple[str, List[str]]]):
        all_queries = [f[0] for f in features]
        all_passages = []
        all_reranker_scores = []
        for f in features:
            all_passages.extend(f[1])
            all_reranker_scores.append(f[2])
        
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

        # Convert reranker scores to tensor
        all_reranker_scores = torch.tensor(all_reranker_scores, dtype=self.torch_dtype)
        return q_collated, d_collated, all_reranker_scores
