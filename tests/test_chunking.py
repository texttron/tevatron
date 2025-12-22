import sys
from pathlib import Path

import pytest


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    # tevatron/tests/test_chunking.py -> tevatron/ -> tevatron/src
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


def _strictly_increasing(xs):
    return all(xs[i] > xs[i - 1] for i in range(1, len(xs)))

REAL_TEXT = (
    "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical "
    "development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging "
    "(MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to "
    "calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in "
    "preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter "
    "development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white "
    "matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to "
    "1.2 microm2/ms. In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both "
    "times were similar (1.2 versus 1.1 microm2/ms). Relative anisotropy was higher the closer birth was to term with "
    "greater absolute values in the internal capsule than in the central white matter. Preterm infants at term showed "
    "higher mean diffusion coefficients in the central white matter (1.4 +/- 0.24 versus 1.15 +/- 0.09 microm2/ms, "
    "p = 0.016) and lower relative anisotropy in both areas compared with full-term infants (white matter, 10.9 +/- "
    "0.6 versus 22.9 +/- 3.0%, p = 0.001; internal capsule, 24.0 +/- 4.44 versus 33.1 +/- 0.6% p = 0.006). "
    "Nonmyelinated fibers in the corpus callosum were visible by diffusion tensor MRI as early as 28 wk; full-term and "
    "preterm infants at term showed marked differences in white matter fiber organization. The data indicate that "
    "quantitative assessment of water diffusion by diffusion tensor MRI provides insight into microstructural "
    "development in cerebral white matter in living infants"
)

TOKENIZER_DIR_NAME = "retriever-qwen3-emb-ft-chunk-batch-2-group-16-maxlen-512-chunk-256-eos"


@pytest.fixture(scope="session")
def train_tokenizer():
    """
    Use the exact tokenizer saved by the finetune_with_chunk.sh output_dir,
    and mimic tevatron.retriever.driver.train's tokenizer setup.
    """
    _add_tevatron_src_to_path()
    from transformers import AutoTokenizer

    tok_dir = _tevatron_root() / TOKENIZER_DIR_NAME
    if not tok_dir.exists():
        pytest.skip(f"local tokenizer dir not found: {tok_dir}")

    tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"  # finetune_with_chunk.sh uses --padding_side right
    return tok


@pytest.mark.unit
def test_train_collator_chunked_passages(train_tokenizer):
    """
    Restore finetune_with_chunk.sh passage chunking scene:
    - passage_max_len=512
    - passage_chunk_size=256
    - pad_to_multiple_of=16 (DataArguments default)
    - padding_side=right
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        passage_max_len=512,
        passage_chunk_size=256,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    
    # ========================================================================
    # NOTE: This test directly calls _tokenize_and_pad_chunked_passages() instead
    #       of collator.__call__() to test chunking in isolation.
    #
    # If we used collator.__call__(features) with passage_chunk_size > 0, it would return:
    #   (q_batch, p_batch, sep_positions)  # 3-element tuple
    #
    # Where:
    #   - q_batch: dict with "input_ids" and "attention_mask" for queries
    #   - p_batch: dict with "input_ids" and "attention_mask" for chunked passages
    #   - sep_positions: list of lists, e.g., [[255, 430]] - EOS token positions per passage
    #                    Used by the model to extract chunk embeddings via MaxSim pooling
    # ========================================================================
    d_collated, sep_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])

    got_ids = d_collated["input_ids"][0].tolist()
    got_mask = d_collated["attention_mask"][0].tolist()
    got_unpadded_len = sum(got_mask)

    assert got_unpadded_len == 431
    assert sep_positions == [[255, 430]]
    # EOS token at sep positions
    assert got_ids[255] == train_tokenizer.eos_token_id
    assert got_ids[430] == train_tokenizer.eos_token_id
    print("length of got_ids: ", len(got_ids))

    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802, 82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279, 9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684, 349, 2326, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77, 284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239, 315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434, 320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622, 4158, 4925, 279, 3076, 9981, 57330, 35606, 518, 220, 17, 23, 73760, 572, 1550, 11, 220, 16, 13, 23, 19197, 441, 17, 58634, 11, 323, 24938, 8841, 4647, 311, 220, 16, 13, 17, 19197, 441, 17, 58634, 13, 758, 279, 44900, 47594, 315, 279, 5306, 47639, 11, 279, 3076, 9981, 57330, 36829, 518, 2176, 3039, 1033, 4428, 320, 16, 13, 17, 19041, 220, 16, 13, 16, 19197, 441, 17, 58634, 568, 39402, 458, 285, 354, 17764, 572, 5080, 279, 12128, 7194, 572, 311, 4647, 448, 7046, 10740, 2750, 304, 279, 5306, 47639, 1091, 304, 279, 8622, 4158, 4925, 13, 4968, 4991, 41434, 518, 4647, 8542, 5080, 3076, 57330, 36829, 304, 279, 8622, 4158, 4925, 320, 16, 13, 19, 51615, 220, 15, 13, 17, 19, 19041, 220, 16, 13, 16, 151645, 20, 51615, 220, 15, 13, 15, 24, 19197, 441, 17, 58634, 11, 281, 284, 220, 15, 13, 15, 16, 21, 8, 323, 4722, 8674, 458, 285, 354, 17764, 304, 2176, 5671, 7707, 448, 2480, 9663, 41434, 320, 5782, 4925, 11, 220, 16, 15, 13, 24, 51615, 220, 15, 13, 21, 19041, 220, 17, 17, 13, 24, 51615, 220, 18, 13, 15, 13384, 281, 284, 220, 15, 13, 15, 15, 16, 26, 5306, 47639, 11, 220, 17, 19, 13, 15, 51615, 220, 19, 13, 19, 19, 19041, 220, 18, 18, 13, 16, 51615, 220, 15, 13, 21, 4, 281, 284, 220, 15, 13, 15, 15, 21, 568, 11581, 2408, 301, 15479, 48674, 304, 279, 42094, 1620, 385, 1242, 1033, 9434, 553, 57330, 15626, 51360, 438, 4124, 438, 220, 17, 23, 73760, 26, 2480, 9663, 323, 855, 4991, 41434, 518, 4647, 8542, 12864, 11799, 304, 4158, 4925, 23788, 7321, 13, 576, 821, 13216, 429, 46516, 15449, 315, 3015, 57330, 553, 57330, 15626, 51360, 5707, 20017, 1119, 8003, 95697, 4401, 304, 59645, 4158, 4925, 304, 5382, 41434, 151645, 151643
    ]
    assert got_ids == expected_ids

    # Hardcoded attention_mask: 431 ones (unpadded tokens) + 1 zero (padding)
    # Padded to multiple of 16: 431 -> 432
    expected_mask = [1] * 431 + [0] * 1
    assert len(got_mask) == 432
    assert got_mask == expected_mask
    # Verify attention_mask is 1 at sep_positions (EOS tokens should be attended)
    assert got_mask[255] == 1
    assert got_mask[430] == 1


@pytest.mark.unit
def test_chunk_size_zero_with_train_tokenizer_disables_chunking_and_truncates(train_tokenizer):
    """
    With passage_chunk_size=0, TrainCollator should take the non-chunk path and
    truncate passages to passage_max_len (like finetune_mldr_dev.sh).
    
    Hardcoded golden output: both passages are truncated to exactly 64 tokens
    (passage_max_len), with no padding needed since 64 is already a multiple of 16.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        query_max_len=32,
        passage_max_len=64,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
        train_group_size=2,
        passage_chunk_size=0,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)

    # ========================================================================
    # HOW features IS CONSTRUCTED:
    # ========================================================================
    # features mimics what TrainDataset.__getitem__() returns. Each element is:
    #   (query_tuple, list_of_passage_tuples)
    #
    # Where:
    #   - query_tuple: (text, image, video, audio) - in this test, only text is used
    #   - list_of_passage_tuples: [(text, image, video, audio), ...] - one per passage
    #
    # Structure breakdown:
    #   - ("q1", None, None, None) = query with text="q1", no multimodal content
    #   - [(REAL_TEXT, ...), (REAL_TEXT, ...)] = 2 passages (train_group_size=2)
    #     Each passage tuple: (text=REAL_TEXT, image=None, video=None, audio=None)
    # ========================================================================
    features = [
        (("q1", None, None, None), [(REAL_TEXT, None, None, None), (REAL_TEXT, None, None, None)]),
    ]

    # ========================================================================
    # WHAT collator(features) RETURNS:
    # ========================================================================
    # Since passage_chunk_size=0 (no chunking), TrainCollator.__call__() returns:
    #   (q_batch, p_batch)  # 2-element tuple
    #
    # Where:
    #   q_batch: dict with PyTorch tensors for queries
    #     - "input_ids": tensor([[token_ids for "q1"]])  # shape: [num_queries, query_seq_len]
    #     - "attention_mask": tensor([[1, 1, ...]])      # shape: [num_queries, query_seq_len]
    #
    #   p_batch: dict with PyTorch tensors for passages (FLATTENED across all queries)
    #     - "input_ids": tensor([
    #         [token_ids for passage 1 (truncated to passage_max_len=64)],
    #         [token_ids for passage 2 (truncated to passage_max_len=64)]
    #       ])  # shape: [total_passages, passage_seq_len]
    #     - "attention_mask": tensor([
    #         [1, 1, ..., 1],  # 64 ones (no padding since 64 is multiple of 16)
    #         [1, 1, ..., 1]   # 64 ones
    #       ])  # shape: [total_passages, passage_seq_len]
    #
    # Note: The collator flattens all passages from all queries into a single batch.
    #       With 1 query and train_group_size=2, we get 2 passages in p_batch.
    # ========================================================================
    out = collator(features)
    assert len(out) == 2  # Verify non-chunking path returns 2 elements
    q_batch, p_batch = out  # Unpack: q_batch (queries), p_batch (passages)

    # Hardcoded golden output (both passages are identical since same input text)
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, 151643
    ]
    expected_mask = [1] * 64  # No padding needed (64 is multiple of 16)

    assert p_batch["input_ids"].shape[0] == 2  # train_group_size=2

    for i in range(p_batch["input_ids"].shape[0]):
        got_ids = p_batch["input_ids"][i].tolist()
        got_mask = p_batch["attention_mask"][i].tolist()
        unpadded_len = sum(got_mask)

        assert unpadded_len == 64
        assert len(got_ids) == 64
        assert len(got_mask) == 64
        assert got_ids == expected_ids
        assert got_mask == expected_mask


@pytest.mark.unit
def test_chunking_chunk_size_equal_maxlen_is_capped_to_single_chunk(train_tokenizer):
    """
    When chunk_size == max_len, chunking should be capped to exactly max_len total tokens
    (incl. EOS), with exactly one EOS at the end.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    long_text = (REAL_TEXT + " ") * 20
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=64,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, sep_positions = collator._tokenize_and_pad_chunked_passages([long_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, 151645
    ]
    expected_sep_positions = [[63]]
    expected_mask = [1] * 64

    assert sum(mask) == 64
    assert len(ids) == 64
    assert sep_positions == expected_sep_positions
    assert ids == expected_ids
    assert ids[63] == 151645
    assert mask == expected_mask
    assert _strictly_increasing(sep_positions[0])


@pytest.mark.unit
def test_chunking_chunk_size_greater_than_maxlen_is_capped_to_single_chunk(train_tokenizer):
    """
    When chunk_size > max_len, chunking should still be capped to exactly max_len total tokens
    (incl. EOS), with exactly one EOS at the end.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    long_text = (REAL_TEXT + " ") * 20
    data_args = DataArguments(
        passage_chunk_size=128,
        passage_max_len=64,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, sep_positions = collator._tokenize_and_pad_chunked_passages([long_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output (same as chunk_size == max_len case)
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, 151645
    ]
    expected_sep_positions = [[63]]
    expected_mask = [1] * 64

    assert sum(mask) == 64
    assert len(ids) == 64
    assert sep_positions == expected_sep_positions
    assert ids == expected_ids
    assert ids[63] == 151645
    assert mask == expected_mask
    assert _strictly_increasing(sep_positions[0])


@pytest.mark.unit
def test_chunking_short_passage_shorter_than_chunk_size(train_tokenizer):
    """
    When passage is shorter than chunk_size, it should still get one chunk with EOS,
    and padding should be applied to pad_to_multiple_of.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    short_text = "Hello world"
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, sep_positions = collator._tokenize_and_pad_chunked_passages([short_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: "Hello world" -> 2 tokens + 1 EOS = 3 tokens, padded to 16
    expected_ids = [9707, 1879, 151645] + [151643] * 13  # 3 content + 13 padding
    expected_sep_positions = [[2]]
    expected_mask = [1, 1, 1] + [0] * 13  # 3 ones + 13 zeros

    assert sum(mask) == 3
    assert len(ids) == 16  # Padded to multiple of 16
    assert sep_positions == expected_sep_positions
    assert ids == expected_ids
    assert ids[2] == 151645  # EOS at position 2
    assert mask == expected_mask
    assert _strictly_increasing(sep_positions[0])


@pytest.mark.unit
def test_chunking_passage_needs_padding_unpadded_not_multiple_of_pad_to_multiple_of(train_tokenizer):
    """
    When unpadded length is not a multiple of pad_to_multiple_of, padding should be added.
    This tests: unpadded_len=50, pad_to_multiple_of=16 -> padded_len=64.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        passage_chunk_size=32,
        passage_max_len=50,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, sep_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: 50 unpadded tokens (2 chunks: 31+1 EOS, 18+1 EOS), padded to 64
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        151645, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629,
        279, 9981, 57330, 151645
    ] + [151643] * 14  # 50 content + 14 padding
    expected_sep_positions = [[31, 49]]
    expected_mask = [1] * 50 + [0] * 14  # 50 ones + 14 zeros

    assert sum(mask) == 50
    assert len(ids) == 64  # Padded to multiple of 16
    assert sep_positions == expected_sep_positions
    assert ids == expected_ids
    assert ids[31] == 151645  # First EOS
    assert ids[49] == 151645  # Second EOS
    assert mask == expected_mask
    assert _strictly_increasing(sep_positions[0])


@pytest.mark.unit
def test_chunking_multiple_passages_different_lengths(train_tokenizer):
    """
    Test batch processing with multiple passages of different lengths:
    - Short passage (2 tokens)
    - Medium passage (18 tokens)
    - Long passage (128 tokens, multiple chunks)
    All should be padded to the same length (longest unpadded length rounded up to pad_to_multiple_of).
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    texts = ["Short", REAL_TEXT[:100], REAL_TEXT]
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, sep_positions = collator._tokenize_and_pad_chunked_passages(texts)
    
    # Hardcoded golden outputs
    # Passage 0: "Short" -> 1 token + 1 EOS = 2 tokens, padded to 128
    expected_ids_0 = [12472, 151645] + [151643] * 126
    expected_mask_0 = [1, 1] + [0] * 126
    expected_sep_0 = [1]
    
    # Passage 1: REAL_TEXT[:100] -> 17 tokens + 1 EOS = 18 tokens, padded to 128
    expected_ids_1 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        1062, 151645
    ] + [151643] * 110
    expected_mask_1 = [1] * 18 + [0] * 110
    expected_sep_1 = [17]
    
    # Passage 2: REAL_TEXT -> 2 chunks (63+1 EOS, 63+1 EOS) = 128 tokens, no padding needed
    expected_ids_2 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, 151645, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77,
        284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239,
        315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434,
        320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622,
        151645
    ]
    expected_mask_2 = [1] * 128
    expected_sep_2 = [63, 127]

    ids_0 = d_collated["input_ids"][0].tolist()
    mask_0 = d_collated["attention_mask"][0].tolist()
    ids_1 = d_collated["input_ids"][1].tolist()
    mask_1 = d_collated["attention_mask"][1].tolist()
    ids_2 = d_collated["input_ids"][2].tolist()
    mask_2 = d_collated["attention_mask"][2].tolist()

    # Passage 0 assertions
    assert sum(mask_0) == 2
    assert len(ids_0) == 128
    assert ids_0 == expected_ids_0
    assert mask_0 == expected_mask_0
    assert sep_positions[0] == expected_sep_0
    
    # Passage 1 assertions
    assert sum(mask_1) == 18
    assert len(ids_1) == 128
    assert ids_1 == expected_ids_1
    assert mask_1 == expected_mask_1
    assert sep_positions[1] == expected_sep_1
    
    # Passage 2 assertions
    assert sum(mask_2) == 128
    assert len(ids_2) == 128
    assert ids_2 == expected_ids_2
    assert mask_2 == expected_mask_2
    assert sep_positions[2] == expected_sep_2
    assert _strictly_increasing(sep_positions[2])

