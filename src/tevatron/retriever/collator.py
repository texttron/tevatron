import logging
import torch
from typing import List, Tuple, Optional
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
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        all_queries = [q[0] for q in all_queries]

        def _get_passage_text(passage):
            if isinstance(passage, dict):
                return passage.get("passage", "")
            return passage[0]

        all_passages = [_get_passage_text(p) for p in all_passages]
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
class ChunkMaxSimTrainCollator:
    """
    Collator that expands passages into chunks and tracks chunk-to-passage mapping
    for MaxSim pooling during training.
    """
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    extra_keys: Optional[List[str]] = None

    def _extract_query_text(self, query_entry):
        if isinstance(query_entry, (tuple, list)):
            return query_entry[0]
        if isinstance(query_entry, dict):
            return query_entry.get("text", "")
        return query_entry

    def _extract_passages(self, feature):
        if isinstance(feature, dict):
            passages = feature.get("passages")
            if passages is not None:
                return passages
            # fallback to already flattened passage with chunks
            passage_chunks = feature.get("passage_chunks")
            passage_text = feature.get("passage", "")
            if passage_chunks is None and passage_text is not None:
                return [{"passage": passage_text, "passage_chunks": [passage_text]}]
            return [{"passage": passage_text, "passage_chunks": passage_chunks or []}]
        # legacy tuple format: (query_tuple, [passage_dict, ...])
        return feature[1]

    def _extract_query(self, feature):
        if isinstance(feature, dict):
            return feature.get("query")
        return feature[0]

    def _get_chunk_list(self, passage):
        if isinstance(passage, dict):
            chunks = passage.get("passage_chunks")
            if chunks and len(chunks) > 0:
                return chunks
            text = passage.get("passage", "")
            return [text]
        if isinstance(passage, (tuple, list)):
            return [passage[0]]
        return [passage]

    def __call__(self, features):
        queries = [self._extract_query_text(self._extract_query(f)) or '' for f in features]

        flat_chunks: List[str] = []
        chunk_to_passage: List[int] = []
        passage_idx = 0

        for feature in features:
            passages = self._extract_passages(feature) or []
            for passage in passages:
                chunk_list = self._get_chunk_list(passage)
                if not chunk_list:
                    chunk_list = ['']
                for chunk in chunk_list:
                    flat_chunks.append(chunk)
                    chunk_to_passage.append(passage_idx)
                passage_idx += 1

        if not flat_chunks:
            raise ValueError("No passage chunks found in batch; ensure dataset provides passage_chunks.")

        q_inputs = self.tokenizer(
            queries,
            padding=False,
            truncation=True,
            max_length=self.data_args.query_max_len - 1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        d_inputs = self.tokenizer(
            flat_chunks,
            padding=False,
            truncation=True,
            max_length=self.data_args.passage_max_len - 1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_inputs['input_ids'] = [ids + [self.tokenizer.eos_token_id] for ids in q_inputs['input_ids']]
            d_inputs['input_ids'] = [ids + [self.tokenizer.eos_token_id] for ids in d_inputs['input_ids']]

        q_inputs = self.tokenizer.pad(
            q_inputs,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        d_inputs = self.tokenizer.pad(
            d_inputs,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        batch = {
            "query_input_ids": q_inputs["input_ids"],
            "query_attention_mask": q_inputs["attention_mask"],
            "passage_input_ids": d_inputs["input_ids"],
            "passage_attention_mask": d_inputs["attention_mask"],
            "chunk_to_passage": torch.tensor(chunk_to_passage, dtype=torch.long),
        }

        if "token_type_ids" in q_inputs:
            batch["query_token_type_ids"] = q_inputs["token_type_ids"]
        if "token_type_ids" in d_inputs:
            batch["passage_token_type_ids"] = d_inputs["token_type_ids"]

        if self.extra_keys:
            for key in self.extra_keys:
                values = []
                for feature in features:
                    if isinstance(feature, dict) and key in feature:
                        values.append(feature[key])
                if values:
                    if isinstance(values[0], torch.Tensor):
                        batch[key] = torch.stack(values)
                    elif isinstance(values[0], (int, float)):
                        dtype = torch.long if isinstance(values[0], int) else torch.float
                        batch[key] = torch.tensor(values, dtype=dtype)
                    else:
                        batch[key] = values

        return batch


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
            passage = all_passages[idx]
            if isinstance(passage, dict):
                text = passage.get("passage")
                image = passage.get("image")
                video = passage.get("video")
                audio = passage.get("audio")
            else:
                text = passage[0]
                image = passage[1]
                video = passage[2]
                audio = passage[3]
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
