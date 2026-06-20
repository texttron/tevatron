import logging

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .bidirectional import make_bidirectional
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class SpladeModel(EncoderModel):
    """SPLADE over a masked-LM backbone (BERT-family). The original Tevatron
    SPLADE; max-pools ``log(1+relu(logits))`` over the sequence."""

    TRANSFORMER_CLS = AutoModelForMaskedLM

    def encode_query(self, qry):
        qry_out = self.encoder(**qry, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(qry_out)) * qry['attention_mask'].unsqueeze(-1), dim=1)
        return aggregated_psg_out

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)


class SpladeModelForCausalLM(EncoderModel):
    """SPLADE over a decoder-only LM backbone (Llama 3 / Qwen2.5).

    This is the LACONIC variant: a causal LM is used as a SPLADE encoder by
    projecting hidden states through its ``lm_head`` to vocab logits, then
    aggregating over the sequence into one sparse vocab-space vector. Two knobs
    distinguish it from the MLM ``SpladeModel``:

    - ``is_bidirectional``: convert the decoder to full (BERT-like) attention so
      every token sees the whole sequence before aggregation. See
      ``bidirectional.make_bidirectional``.
    - ``pooling_strategy`` over the sequence:
        ``max``  — SPLADE max-pool of ``log(1+relu(logit))`` (default);
        ``mean`` — masked mean of ``log(1+relu(logit))``;
        ``last`` — ``log(1+relu(logit))`` at the last non-pad token.

    Same ``EncoderModel`` contract (q/p reps, dot-product similarity, listwise
    CE) — only the encode path changes, so the existing trainer / collator /
    sparse-encode path work unchanged. FLOPS sparsity regularization lives in
    ``SpladeTrainer`` (training-only), not here.
    """

    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self, *args, pooling_strategy: str = "max", **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_strategy = pooling_strategy

    def _aggregate(self, logits, attention_mask):
        """(bsz, seq, vocab) logits + (bsz, seq) mask -> (bsz, vocab) sparse rep."""
        mask = attention_mask.unsqueeze(-1)  # (bsz, seq, 1)
        if self.pooling_strategy == "max":
            # mask padding to -inf before relu so it never wins the max
            masked = logits + (1 - mask) * -1e4
            pooled, _ = torch.max(masked, dim=1)
            return torch.log1p(torch.relu(pooled))
        elif self.pooling_strategy == "mean":
            activated = torch.log1p(torch.relu(logits)) * mask
            return activated.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling_strategy == "last":
            seq_len = attention_mask.sum(dim=1) - 1  # last non-pad index
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_idx, seq_len, :]
            return torch.log1p(torch.relu(last_logits))
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy!r}")

    def encode_query(self, qry):
        if qry is None:
            return None
        out = self.encoder(**qry, return_dict=True).logits
        return self._aggregate(out, qry["attention_mask"])

    def encode_passage(self, psg):
        # passage is encoded identically to query
        return self.encode_query(psg)

    # --- construction --------------------------------------------------------
    # We override build/load (rather than reuse EncoderModel's) only to: pick
    # AutoModelForCausalLM, optionally flip the backbone bidirectional *before*
    # any LoRA wrap, and thread ``pooling_strategy`` through. Everything else
    # mirrors the parent.

    @classmethod
    def build(cls, model_args, train_args, **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        if getattr(model_args, "is_bidirectional", False):
            base_model = make_bidirectional(base_model)

        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
            encoder = lora_model
        else:
            encoder = base_model

        return cls(
            encoder=encoder,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
            pooling_strategy=getattr(model_args, "pooling_strategy", "max"),
        )

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             is_bidirectional: bool = False,
             pooling_strategy: str = "max",
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        if is_bidirectional:
            base_model = make_bidirectional(base_model)

        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            encoder = lora_model.merge_and_unload()
        else:
            encoder = base_model

        return cls(
            encoder=encoder,
            pooling=pooling,
            normalize=normalize,
            pooling_strategy=pooling_strategy,
        )
