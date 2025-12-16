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
            # If training with chunked passages, sep_positions is produced by the collator and
            # attached to the model by TevatronTrainer.compute_loss(). Forward() needs to pass it
            # into encode_passage() to actually get chunk reps/masks.
            sep_positions = getattr(self, "sep_positions", None)
            if self.passage_chunk_size > 0 and sep_positions is not None:
                print(f"sep_positions: {sep_positions}")
                try:
                    p_reps = self.encode_passage(passage, sep_positions=sep_positions)
                except TypeError:
                    # Some models (e.g., multimodal) don't accept sep_positions.
                    p_reps = self.encode_passage(passage)
            else:
                p_reps = self.encode_passage(passage)
            print(f"p_reps: {p_reps}")
            print(f"type(p_reps): {type(p_reps)}")
            if self.passage_chunk_size > 0 and isinstance(p_reps, tuple):
                p_reps, chunk_mask = p_reps

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            print(f"passage_chunk_size: {self.passage_chunk_size}")
            print(f"chunk_mask: {chunk_mask}")
            if self.passage_chunk_size > 0 and chunk_mask is not None:
                print(f"start compute maxsim similarity==========================")
                scores = self.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
                print(f"end compute maxsim similarity==========================")
            else:
                print(f"start compute similarity==========================")
                scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            num_psg_per_query = scores.size(1) // q_reps.size(0)
            target = torch.arange(q_reps.size(0), device=scores.device, dtype=torch.long)
            target = target * num_psg_per_query

            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size # counter average weight reduction
        # for eval
        else:
            if self.passage_chunk_size > 0 and chunk_mask is not None:
                scores = self.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
            else:
                scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
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
        Q: number of queries
        P: number of passages
        C: number of chunks per passage
        H: dimension of the embeddings
        Returns: [Q, P]
        """
        chunk_scores = torch.einsum('qh,pch->qpc', q_reps, p_reps) # 第 q 个 query 和第 p 个 passage 的第 c 个 chunk 的相似度
        if chunk_mask is not None:
            padding_mask = ~chunk_mask.unsqueeze(0).bool()
            chunk_scores = chunk_scores.masked_fill(padding_mask, float('-inf'))
        max_vals, max_idx = chunk_scores.max(dim=-1)  # [Q, P], [Q, P]

        # Print argmax chunk index + (optional) original token position from sep_positions
        if True:
            # only log from rank-0 if DDP
            if (not getattr(self, "is_ddp", False)) or getattr(self, "process_rank", 0) == 0:
                sep_positions = getattr(self, "sep_positions", None)
                # If DDP gathered passages, sep_positions may not align; only use when sizes match.
                sep_ok = (
                    isinstance(sep_positions, (list, tuple))
                    and len(sep_positions) == p_reps.size(0)
                )
                qn, pn = max_idx.size(0), max_idx.size(1)
                for qi in range(qn):
                    for pi in range(pn):
                        ci = int(max_idx[qi, pi].item())
                        # last valid chunk index for this passage (by mask)
                        if chunk_mask is not None:
                            valid = int(chunk_mask[pi].sum().item())
                            last_ci = max(valid - 1, 0)
                        else:
                            last_ci = p_reps.size(1) - 1

                        if sep_ok and sep_positions[pi]:
                            pos_list = sep_positions[pi]
                            best_pos = pos_list[ci] if 0 <= ci < len(pos_list) else None
                            last_pos = pos_list[-1]
                            logger.info(
                                f"[maxsim] q={qi} p={pi} best_chunk={ci} best_pos={best_pos} "
                                f"last_chunk={last_ci} last_pos={last_pos} best_score={float(max_vals[qi, pi].item()):.6f}"
                            )
                        else:
                            logger.info(
                                f"[maxsim] q={qi} p={pi} best_chunk={ci} last_chunk={last_ci} "
                                f"best_score={float(max_vals[qi, pi].item()):.6f}"
                            )

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
