import logging
import random
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from qwen_omni_utils import process_mm_info
from PIL import Image
from rich import print

from tevatron.retriever.arguments import DataArguments
torch.set_printoptions(threshold=float('inf'), linewidth=10000)

logger = logging.getLogger(__name__)


def _chunk_tokens(
    tokens: List[int],
    chunk_size: int,
    eos_token_id: int,
    max_length: int = None,
    chunk_size_range: Optional[Tuple[int, int]] = None,
    passage_index: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Chunk tokens into chunks with EOS separators.

    :param tokens: Token IDs to chunk
    :param chunk_size: Fixed chunk size (before EOS). Must be >= 2. Used when chunk_size_range is None.
    :param eos_token_id: EOS token ID to append after each chunk
    :param max_length: Optional max total length (including EOS). If None, no limit.
    :param chunk_size_range: Optional (min, max) tuple for variable chunk sizes. If set, each chunk uses a random size in [min, max].
    :param passage_index: Index of the passage in the batch (for DDP-consistent random seeds)
    :return: (chunked_ids, eos_positions) - token IDs with EOS separators and EOS positions
    """
    # Validate and set up chunk size parameters
    if chunk_size_range:
        chunk_size_min, chunk_size_max = chunk_size_range
        use_variable_sizes = True
    else:
        if chunk_size < 2:
            return [], []
        use_variable_sizes = False

    # Chunk tokens and add EOS after each chunk
    ids = []
    eos_pos = []
    i = 0
    total_length = 0
    chunk_index = 0

    while i < len(tokens):
        # Pick chunk size for this chunk
        if use_variable_sizes:
            # DDP-safe: Use deterministic seed based on passage and chunk index
            # This ensures all ranks generate the same chunk sizes for the same passage
            seed = hash((passage_index, chunk_index)) & 0xFFFFFFFF
            rng = random.Random(seed)
            current_chunk_size = rng.randint(chunk_size_min, chunk_size_max)
            chunk_index += 1
        else:
            current_chunk_size = chunk_size
        
        # Check if we would exceed max_length with this chunk
        if max_length and total_length + current_chunk_size > max_length:
            # Use remaining space (leave 1 for EOS if possible)
            remaining = max_length - total_length - 1
            if remaining > 0:
                take = min(remaining, len(tokens) - i)
                ids.extend(tokens[i:i + take])
                ids.append(eos_token_id)
                eos_pos.append(len(ids) - 1)
            break
        
        # Take tokens for this chunk (reserve 1 slot for EOS)
        current_chunk_len = current_chunk_size - 1
        take = min(current_chunk_len, len(tokens) - i)
        ids.extend(tokens[i:i + take])
        ids.append(eos_token_id)
        eos_pos.append(len(ids) - 1)
        
        total_length += take + 1  # +1 for EOS
        i += take
    
    return ids, eos_pos

def _pad_and_adjust_eos_positions(
    all_input_ids: List[List[int]],
    all_eos_positions: List[List[int]],
    tokenizer: PreTrainedTokenizer,
    padding_side: str,
    pad_to_multiple_of: int,
) -> Tuple[dict, List[List[int]]]:
    """
    Pad input IDs and adjust EOS positions for left padding.
    
    :param all_input_ids: List of token ID lists (one per passage)
    :param all_eos_positions: List of EOS position lists (one per passage)
    :param tokenizer: Tokenizer for padding
    :param padding_side: 'left' or 'right'
    :param pad_to_multiple_of: Pad to multiple of this value
    :return: (padded_dict, adjusted_eos_positions) - padded tensors and adjusted EOS positions
    """
    d_collated = {'input_ids': all_input_ids}
    original_lengths = [len(ids) for ids in all_input_ids]
    tokenizer.padding_side = padding_side
    
    d_collated = tokenizer.pad(
        d_collated,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Shift EOS positions for left padding
    adjusted_eos_positions = [list(eos_pos_list) for eos_pos_list in all_eos_positions]
    if padding_side == 'left':
        padded_lengths = d_collated['input_ids'].shape[1]
        for i, eos_pos_list in enumerate(adjusted_eos_positions):
            padding_length = padded_lengths - original_lengths[i]
            adjusted_eos_positions[i] = [pos + padding_length for pos in eos_pos_list]
    
    return d_collated, adjusted_eos_positions


def _tokenize_and_pad_chunked_passages(
    passages: List[str],
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    chunk_sizes: Optional[List[int]] = None,
    chunk_size_range: Optional[Tuple[int, int]] = None,
) -> Tuple[dict, List[List[int]]]:
    """
    Tokenize and chunk passages with EOS separators. Each chunk ends with EOS for embedding extraction.
    
    :param passages: Passage texts to tokenize and chunk
    :param tokenizer: Tokenizer for encoding
    :param data_args: DataArguments with chunk_size, max_len, pad_to_multiple_of
    :param chunk_sizes: Optional list of chunk sizes (one per passage). If None, uses data_args.passage_chunk_size
    :param chunk_size_range: Optional (min, max) tuple for variable chunk sizes per chunk. If set, each chunk within a passage uses a random size.
    :return: (collated_dict, eos_positions) - padded tensors and EOS positions per passage
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None; cannot chunk passages with EOS separators.")
    if chunk_sizes is not None and len(chunk_sizes) != len(passages):
        raise ValueError(f"chunk_sizes length ({len(chunk_sizes)}) must match passages length ({len(passages)})")
    max_length = data_args.passage_max_len  # cap total length (incl. EOS per chunk)
    
    all_input_ids = []
    all_eos_positions = []
    
    for idx, passage in enumerate(passages):
        if passage is None:
            passage = ""
        tokens = tokenizer.encode(passage, add_special_tokens=False)
        # Use per-passage chunk size if provided, otherwise use fixed chunk size
        # Note: chunk_size is ignored in _chunk_tokens when chunk_size_range is provided
        chunk_size = chunk_sizes[idx] if chunk_sizes is not None else data_args.passage_chunk_size
        ids, eos_pos = _chunk_tokens(
            tokens=tokens,
            chunk_size=chunk_size,
            eos_token_id=eos_id,
            max_length=max_length,
            chunk_size_range=chunk_size_range,
            passage_index=idx,
        )
        all_input_ids.append(ids)
        all_eos_positions.append(eos_pos)
    
    d_collated, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=tokenizer,
        padding_side=data_args.padding_side,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )
    
    return d_collated, adjusted_eos_positions


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
        
        # Check if we should use chunking (fixed or random)
        use_fixed_chunking = self.data_args.passage_chunk_size > 0
        
        if self.data_args.passage_chunk_size_range is not None:
            # Parse range string (e.g., "64, 128" or "64,128")
            try:
                parts = [p.strip() for p in self.data_args.passage_chunk_size_range.split(',')]
                if len(parts) != 2:
                    raise ValueError(f"passage_chunk_size_range must contain exactly 2 values separated by comma, got: {self.data_args.passage_chunk_size_range}")
                chunk_size_min = int(parts[0])
                chunk_size_max = int(parts[1])
            except ValueError as e:
                raise ValueError(f"Invalid passage_chunk_size_range format '{self.data_args.passage_chunk_size_range}'. Expected format: 'min,max' (e.g., '64,128')") from e
            
            # Validate range
            if chunk_size_min < 2:
                raise ValueError(f"Minimum chunk size must be >= 2, got {chunk_size_min}")
            if chunk_size_max < chunk_size_min:
                raise ValueError(f"Maximum chunk size ({chunk_size_max}) must be >= minimum chunk size ({chunk_size_min})")
            
            if self.data_args.passage_chunk_size_variable:
                # Variable chunk sizes: each chunk within a passage gets a random size
                # Pass the range to _chunk_tokens, which will randomly pick a size for each chunk
                chunk_size_range = (chunk_size_min, chunk_size_max)
                d_collated, eos_positions = self._tokenize_and_pad_chunked_passages(all_passages, chunk_size_range=chunk_size_range)
            else:
                # Fixed random chunk size per passage: all chunks in a passage use the same random size
                # Generate random chunk sizes for each passage
                chunk_sizes = [random.randint(chunk_size_min, chunk_size_max) for _ in all_passages]
                d_collated, eos_positions = self._tokenize_and_pad_chunked_passages(all_passages, chunk_sizes=chunk_sizes)
            return q_collated, d_collated, eos_positions
        elif use_fixed_chunking:
            d_collated, eos_positions = self._tokenize_and_pad_chunked_passages(all_passages)
            # DDP Safety: Ensure eos_positions is always returned when chunking is enabled
            assert eos_positions is not None, "passage_chunk_size > 0 but eos_positions is None"
            return q_collated, d_collated, eos_positions
        else:
            # No chunking - return without eos_positions
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

    def _tokenize_and_pad_chunked_passages(self, passages: List[str], chunk_sizes: Optional[List[int]] = None, chunk_size_range: Optional[Tuple[int, int]] = None):
        return _tokenize_and_pad_chunked_passages(passages, self.tokenizer, self.data_args, chunk_sizes=chunk_sizes, chunk_size_range=chunk_size_range)


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
class ChunkedEncodeCollator:
    """Collator for chunked passage encoding (inference/search). Supports fixed or random chunk sizes."""
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        """
        Collate chunked passage encoding features.
        :param features: List of (doc_id, text, image, video, audio) tuples
        :return: (doc_ids, collated_inputs, eos_positions)
        """
        doc_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        
        # Check if we should use random chunking
        if self.data_args.passage_chunk_size_range is not None:
            # Parse range string (e.g., "64, 128" or "64,128")
            try:
                parts = [p.strip() for p in self.data_args.passage_chunk_size_range.split(',')]
                if len(parts) != 2:
                    raise ValueError(f"passage_chunk_size_range must contain exactly 2 values separated by comma, got: {self.data_args.passage_chunk_size_range}")
                chunk_size_min = int(parts[0])
                chunk_size_max = int(parts[1])
            except ValueError as e:
                raise ValueError(f"Invalid passage_chunk_size_range format '{self.data_args.passage_chunk_size_range}'. Expected format: 'min,max' (e.g., '64,128')") from e
            
            # Validate range
            if chunk_size_min < 2:
                raise ValueError(f"Minimum chunk size must be >= 2, got {chunk_size_min}")
            if chunk_size_max < chunk_size_min:
                raise ValueError(f"Maximum chunk size ({chunk_size_max}) must be >= minimum chunk size ({chunk_size_min})")
            
            if self.data_args.passage_chunk_size_variable:
                # Variable chunk sizes: each chunk within a passage gets a random size
                chunk_size_range = (chunk_size_min, chunk_size_max)
                d_collated, all_eos_positions = self._tokenize_and_pad_chunked_passages(texts, chunk_size_range=chunk_size_range)
            else:
                # Fixed random chunk size per passage: all chunks in a passage use the same random size
                # Generate random chunk sizes for each passage
                chunk_sizes = [random.randint(chunk_size_min, chunk_size_max) for _ in texts]
                d_collated, all_eos_positions = self._tokenize_and_pad_chunked_passages(texts, chunk_sizes=chunk_sizes)
        else:
            # Use fixed chunking for inference
            d_collated, all_eos_positions = self._tokenize_and_pad_chunked_passages(texts)
        
        return doc_ids, d_collated, all_eos_positions

    def _tokenize_and_pad_chunked_passages(self, passages: List[str], chunk_sizes: Optional[List[int]] = None, chunk_size_range: Optional[Tuple[int, int]] = None):
        return _tokenize_and_pad_chunked_passages(passages, self.tokenizer, self.data_args, chunk_sizes=chunk_sizes, chunk_size_range=chunk_size_range)


@dataclass
class PreChunkedEncodeCollator:
    """
    Collator for pre-chunked passage encoding (inference/search).
    Expects passages as lists of pre-chunked strings and adds EOS tokens between chunks.
    """
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        """
        Collate pre-chunked passage encoding features.
        :param features: List of (doc_id, chunks_list, image, video, audio) tuples
                        where chunks_list is a list of pre-chunked passage strings
        :return: (doc_ids, collated_inputs, eos_positions)
        """
        doc_ids = [x[0] for x in features]
        chunks_lists = [x[1] for x in features]  # List of lists of strings
        
        # Process pre-chunked passages: tokenize each chunk and add EOS between them
        d_collated, all_eos_positions = self._tokenize_and_pad_pre_chunked_passages(chunks_lists)
        
        return doc_ids, d_collated, all_eos_positions

    def _tokenize_and_pad_pre_chunked_passages(self, chunks_lists: List[List[str]]):
        """
        Tokenize pre-chunked passages and add EOS tokens between chunks.
        
        This is used when you have pre-chunked passages (e.g., from ChatGPT or manual chunking).
        Each chunk is tokenized separately, and EOS tokens are inserted between chunks.
        
        :param chunks_lists: List of lists, where each inner list contains pre-chunked passage strings
                            Example: [["chunk1", "chunk2"], ["chunk3"]] for 2 passages
        :return: (collated_dict, eos_positions) - padded tensors and EOS positions per passage
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None; cannot add EOS tokens between chunks.")
        
        max_length = self.data_args.passage_max_len
        all_input_ids = []
        all_eos_positions = []
        
        for chunks in chunks_lists:
            if chunks is None:
                chunks = []
            if not isinstance(chunks, list):
                raise ValueError(f"Expected list of chunks, got {type(chunks)}")
            if len(chunks) == 0:
                # Empty chunks list - create empty passage with no EOS positions
                all_input_ids.append([])
                all_eos_positions.append([])
                continue
            
            # Tokenize each chunk and concatenate with EOS between them
            ids = []
            eos_pos = []
            total_length = 0
            
            for chunk_idx, chunk in enumerate(chunks):
                if chunk is None:
                    chunk = ""
                # Tokenize this chunk (without special tokens, we'll add EOS manually)
                chunk_tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
                
                # Check if adding this chunk + EOS would exceed max_length
                chunk_size = len(chunk_tokens)
                if max_length and total_length + chunk_size + 1 > max_length:
                    # Use remaining space (leave 1 for EOS if possible)
                    remaining = max_length - total_length - 1
                    if remaining > 0:
                        chunk_tokens = chunk_tokens[:remaining]
                        ids.extend(chunk_tokens)
                        ids.append(eos_id)
                        eos_pos.append(len(ids) - 1)
                    break
                
                # Add chunk tokens
                ids.extend(chunk_tokens)
                # Add EOS after each chunk
                ids.append(eos_id)
                eos_pos.append(len(ids) - 1)
                total_length += chunk_size + 1
            
            all_input_ids.append(ids)
            all_eos_positions.append(eos_pos)
        
        d_collated, adjusted_eos_positions = _pad_and_adjust_eos_positions(
            all_input_ids=all_input_ids,
            all_eos_positions=all_eos_positions,
            tokenizer=self.tokenizer,
            padding_side=self.data_args.padding_side,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
        )
        
        return d_collated, adjusted_eos_positions


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
