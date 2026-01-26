from dataclasses import dataclass
from typing import Dict, Optional

import os
import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    chunk_mask: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.passage_chunk_size = 0
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps, chunk_mask = None, None
        if passage:
            # If training with chunked passages, eos_positions is produced by the collator and
            # attached to the model by TevatronTrainer.compute_loss(). Forward() needs to pass it
            # into encode_passage() to actually get chunk reps/masks.
            eos_positions = getattr(self, "eos_positions", None)
            
            # Detailed logging for debugging
            logger.info(f"[Encoder.forward] passage_chunk_size={self.passage_chunk_size}, "
                       f"eos_positions={'None' if eos_positions is None else f'list[{len(eos_positions)}]'}, "
                       f"training={self.training}")
            
            if self.passage_chunk_size > 0 and eos_positions is not None:
                logger.info(f"[Encoder.forward] Calling encode_passage WITH eos_positions")
                try:
                    p_reps = self.encode_passage(passage, eos_positions=eos_positions)
                except TypeError:
                    # Some models (e.g., multimodal) don't accept eos_positions.
                    logger.warning(f"[Encoder.forward] encode_passage doesn't accept eos_positions, calling without")
                    p_reps = self.encode_passage(passage)
            else:
                logger.info(f"[Encoder.forward] Calling encode_passage WITHOUT eos_positions "
                           f"(chunk_size={self.passage_chunk_size}, eos_pos={'exists' if eos_positions else 'None'})")
                p_reps = self.encode_passage(passage)
            
            # Check if we got chunked output
            if self.passage_chunk_size > 0 and isinstance(p_reps, tuple):
                p_reps, chunk_mask = p_reps
                logger.info(f"[Encoder.forward] Got chunked output: p_reps {p_reps.shape}, chunk_mask {chunk_mask.shape}")
            else:
                logger.info(f"[Encoder.forward] Got regular output: p_reps {p_reps.shape if isinstance(p_reps, torch.Tensor) else type(p_reps)}")

        # Return embeddings only during training - loss is computed in trainer
        if q_reps is None or p_reps is None or self.training:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps,
                chunk_mask=chunk_mask
            )

        # for eval/inference
        if self.passage_chunk_size > 0 and chunk_mask is not None:
            scores = self.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
        else:
            scores = self.compute_similarity(q_reps, p_reps)
        return EncoderOutput(
            loss=None,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_maxsim_similarity(self, q_reps, p_reps, chunk_mask):
        """
        MaxSim: max similarity between query and passage chunks.
        q_reps: [Q, H], p_reps: [P, C, H], chunk_mask: [P, C]
        Q: number of queries (total across all ranks after DDP gather)
        P: number of passages (total across all ranks after DDP gather)
        C: number of chunks per passage
        H: dimension of the embeddings
        Returns: [Q, P] - similarity matrix for ALL queries and ALL passages
        """
        Q, H = q_reps.shape
        P, C, _ = p_reps.shape
        
        # Log shapes to verify we're using all gathered passages
        if getattr(self, "is_ddp", False) and getattr(self, "process_rank", 0) == 0:
            logger.info(f"[MaxSim] Computing scores for Q={Q} queries, P={P} passages, C={C} chunks, H={H} dims")
            if self.world_size > 1:
                logger.info(f"[MaxSim] This includes passages from all {self.world_size} ranks (gathered via DDP)")
        
        # Compute similarity: for each query and passage, compute similarity to all chunks
        chunk_scores = torch.einsum('qh,pch->qpc', q_reps, p_reps) # 第 q 个 query 和第 p 个 passage 的第 c 个 chunk 的相似度
        
        if chunk_mask is not None:
            # Mask out padding chunks by setting their scores to -inf
            padding_mask = ~chunk_mask.unsqueeze(0).bool()  # [1, P, C]
            chunk_scores = chunk_scores.masked_fill(padding_mask, float('-inf'))
            
            # Log masking info
            if getattr(self, "is_ddp", False) and getattr(self, "process_rank", 0) == 0:
                valid_chunks = chunk_mask.sum().item()
                total_chunks = chunk_mask.numel()
                logger.info(f"[MaxSim] Masked {total_chunks - valid_chunks}/{total_chunks} padding chunks")
        
        # Take max over chunks for each query-passage pair
        max_vals, max_idx = chunk_scores.max(dim=-1)  # [Q, P], [Q, P]

        # # Log maxsim info: read chunk indices directly from max_idx
        # if True:
        #     # only log from rank-0 if DDP
        #     if (not getattr(self, "is_ddp", False)) or getattr(self, "process_rank", 0) == 0:
        #         eos_positions = getattr(self, "eos_positions", None)
        #         eos_ok = (
        #             isinstance(eos_positions, (list, tuple))
        #             and len(eos_positions) == p_reps.size(0)
        #         )
                
        #         # Compute last valid chunk indices for all passages
        #         if chunk_mask is not None:
        #             last_ci_per_passage = (chunk_mask.sum(dim=1) - 1).clamp(min=0)  # [P]
        #         else:
        #             last_ci_per_passage = torch.full((p_reps.size(0),), p_reps.size(1) - 1, dtype=torch.long)
                
        #         # Log for each query-passage pair
        #         for qi in range(max_idx.size(0)):
        #             for pi in range(max_idx.size(1)):
        #                 ci = int(max_idx[qi, pi].item())  # best chunk index from max_idx
        #                 last_ci = int(last_ci_per_passage[pi].item())
        #                 score = float(max_vals[qi, pi].item())
                        
        #                 if eos_ok and eos_positions[pi] and ci < len(eos_positions[pi]):
        #                     best_pos = eos_positions[pi][ci]
        #                     last_pos = eos_positions[pi][-1]
        #                     logger.info(
        #                         f"[maxsim] q={qi} p={pi} best_chunk={ci} best_pos={best_pos} "
        #                         f"last_chunk={last_ci} last_pos={last_pos} best_score={score:.6f}"
        #                     )
        #                 else:
        #                     logger.info(
        #                         f"[maxsim] q={qi} p={pi} best_chunk={ci} last_chunk={last_ci} "
        #                         f"best_score={score:.6f}"
        #                     )

        return max_vals

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
