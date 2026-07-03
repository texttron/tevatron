"""Turn a decoder LM bidirectional, in place, for causal-LM SPLADE.

A decoder-only LM (Llama 3, Qwen2.5) attends causally: token *i* only sees
tokens ``<= i``. For a *retrieval encoder* that is the wrong inductive bias —
we want every token to see the whole sequence (like BERT) before we max-pool
the vocabulary logits into a sparse representation. LACONIC (following LLM2Vec)
gets this by flipping the attention mask from causal to bidirectional.

The legacy LACONIC repo vendored the whole ``llm2vec`` package (~1.1k LoC over
four model files) to do this, because in transformers <= 4.51 the causal mask
was built inside ``LlamaModel._update_causal_mask`` — a method you had to copy
and patch per architecture. **transformers 5.x removed that method**: the model
``forward`` now calls the standalone ``create_causal_mask(...)`` helper, and
there is a sibling ``create_bidirectional_mask(...)`` with the same signature.

So the dependency collapses to two surgical edits on an *already-loaded* model
(no custom ``from_pretrained``, no weight-key remapping — bidirectional and
causal share identical parameters):

1. Make ``forward`` build a bidirectional mask. The trunk's ``forward`` closes
   over the module-level ``create_causal_mask``; we wrap the bound method so
   that, for the duration of the call, that name resolves to
   ``create_bidirectional_mask``.
2. Clear ``is_causal`` on every attention submodule, so the flash-attention /
   SDPA fast paths (which apply a causal mask from their own ``is_causal`` flag
   and ignore a passed-in mask) don't re-impose causality.

Scope: Llama 3 and Qwen2.5 — the two architectures LACONIC validated. Other
decoders that use the 5.x ``create_causal_mask`` helper would work too; we gate
on ``model_type`` so an unvalidated arch fails loudly rather than silently
training with the wrong mask.
"""

import importlib
import logging
import types

logger = logging.getLogger(__name__)

# config.model_type values we've validated. The conversion mechanism itself is
# architecture-agnostic; this set just keeps us honest about what's tested.
SUPPORTED_MODEL_TYPES = {"llama", "qwen2"}

_FLAG = "_tevatron_bidirectional"  # idempotency marker set on a converted trunk


def make_bidirectional(model):
    """Convert a loaded ``*ForCausalLM`` to bidirectional attention, in place.

    ``model`` is a HF causal-LM (token trunk at ``model.model``, vocab head at
    ``model.lm_head``). Returns the same object. Idempotent.

    Raises ``ValueError`` for architectures we haven't validated, so a typo or
    an unsupported backbone surfaces immediately instead of silently training
    with a causal mask.
    """
    model_type = getattr(model.config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Bidirectional attention is validated for {sorted(SUPPORTED_MODEL_TYPES)}; "
            f"got model_type={model_type!r}. Add it to SUPPORTED_MODEL_TYPES in "
            f"tevatron.retriever.modeling.bidirectional after testing, or train "
            f"with bidirectional disabled."
        )

    # Resolved here (not at import) so the module stays importable on
    # transformers < 5.x, which has no `masking_utils` (the < 5.x bidirectional
    # path is llm2vec-style and lives in the LACONIC repro env, not here).
    try:
        from transformers.masking_utils import create_bidirectional_mask
    except ImportError as e:
        raise ImportError(
            "make_bidirectional requires transformers>=5.x (masking_utils with "
            "create_bidirectional_mask). On older transformers use the llm2vec-style "
            "bidirectional path."
        ) from e

    trunk = model.model  # the decoder trunk (LlamaModel / Qwen2Model)
    if getattr(trunk, _FLAG, False):
        return model  # already converted

    # The module that defines the trunk's `forward` and its `create_causal_mask`
    # global (e.g. transformers.models.llama.modeling_llama).
    model_module = importlib.import_module(type(trunk).__module__)
    orig_forward = trunk.forward  # stock bound forward

    def bidirectional_forward(self, *args, **kwargs):
        saved = model_module.create_causal_mask
        model_module.create_causal_mask = create_bidirectional_mask
        try:
            return orig_forward(*args, **kwargs)
        finally:
            model_module.create_causal_mask = saved

    trunk.forward = types.MethodType(bidirectional_forward, trunk)

    # Clear the kernel-level causal flag on every attention submodule.
    n = 0
    for sub in trunk.modules():
        if getattr(sub, "is_causal", False):
            sub.is_causal = False
            n += 1

    setattr(trunk, _FLAG, True)
    logger.info(
        "Converted %s to bidirectional attention (cleared is_causal on %d modules)",
        model_type, n,
    )
    return model
