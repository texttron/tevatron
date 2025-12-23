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
EOS_TOKEN_ID = 151645
PADDING_TOKEN_ID = 151643

@pytest.fixture(scope="session")
def train_tokenizer():
    """
    Use the Qwen 0.6B tokenizer.
    """
    _add_tevatron_src_to_path()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
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
    #   (q_batch, p_batch, eos_positions)  # 3-element tuple
    #
    # Where:
    #   - q_batch: dict with "input_ids" and "attention_mask" for queries
    #   - p_batch: dict with "input_ids" and "attention_mask" for chunked passages
    #   - eos_positions: list of lists, e.g., [[255, 430]] - EOS token positions per passage
    #                    Used by the model to extract chunk embeddings via MaxSim pooling
    # ========================================================================
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])

    got_ids = d_collated["input_ids"][0].tolist()
    got_mask = d_collated["attention_mask"][0].tolist()
    got_unpadded_len = sum(got_mask)

    assert got_unpadded_len == 431
    assert eos_positions == [[255, 430]]
    # EOS token at eos positions
    assert got_ids[255] == train_tokenizer.eos_token_id
    assert got_ids[430] == train_tokenizer.eos_token_id
    print("length of got_ids: ", len(got_ids))

    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802, 82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279, 9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684, 349, 2326, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77, 284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239, 315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434, 320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622, 4158, 4925, 279, 3076, 9981, 57330, 35606, 518, 220, 17, 23, 73760, 572, 1550, 11, 220, 16, 13, 23, 19197, 441, 17, 58634, 11, 323, 24938, 8841, 4647, 311, 220, 16, 13, 17, 19197, 441, 17, 58634, 13, 758, 279, 44900, 47594, 315, 279, 5306, 47639, 11, 279, 3076, 9981, 57330, 36829, 518, 2176, 3039, 1033, 4428, 320, 16, 13, 17, 19041, 220, 16, 13, 16, 19197, 441, 17, 58634, 568, 39402, 458, 285, 354, 17764, 572, 5080, 279, 12128, 7194, 572, 311, 4647, 448, 7046, 10740, 2750, 304, 279, 5306, 47639, 1091, 304, 279, 8622, 4158, 4925, 13, 4968, 4991, 41434, 518, 4647, 8542, 5080, 3076, 57330, 36829, 304, 279, 8622, 4158, 4925, 320, 16, 13, 19, 51615, 220, 15, 13, 17, 19, 19041, 220, 16, 13, 16, EOS_TOKEN_ID, 20, 51615, 220, 15, 13, 15, 24, 19197, 441, 17, 58634, 11, 281, 284, 220, 15, 13, 15, 16, 21, 8, 323, 4722, 8674, 458, 285, 354, 17764, 304, 2176, 5671, 7707, 448, 2480, 9663, 41434, 320, 5782, 4925, 11, 220, 16, 15, 13, 24, 51615, 220, 15, 13, 21, 19041, 220, 17, 17, 13, 24, 51615, 220, 18, 13, 15, 13384, 281, 284, 220, 15, 13, 15, 15, 16, 26, 5306, 47639, 11, 220, 17, 19, 13, 15, 51615, 220, 19, 13, 19, 19, 19041, 220, 18, 18, 13, 16, 51615, 220, 15, 13, 21, 4, 281, 284, 220, 15, 13, 15, 15, 21, 568, 11581, 2408, 301, 15479, 48674, 304, 279, 42094, 1620, 385, 1242, 1033, 9434, 553, 57330, 15626, 51360, 438, 4124, 438, 220, 17, 23, 73760, 26, 2480, 9663, 323, 855, 4991, 41434, 518, 4647, 8542, 12864, 11799, 304, 4158, 4925, 23788, 7321, 13, 576, 821, 13216, 429, 46516, 15449, 315, 3015, 57330, 553, 57330, 15626, 51360, 5707, 20017, 1119, 8003, 95697, 4401, 304, 59645, 4158, 4925, 304, 5382, 41434, EOS_TOKEN_ID, PADDING_TOKEN_ID
    ]
    assert got_ids == expected_ids

    # Hardcoded attention_mask: 431 ones (unpadded tokens) + 1 zero (padding)
    # Padded to multiple of 16: 431 -> 432
    expected_mask = [1] * 431 + [0] * 1
    assert len(got_mask) == 432
    assert got_mask == expected_mask
    # Verify attention_mask is 1 at eos_positions (EOS tokens should be attended)
    assert got_mask[255] == 1
    assert got_mask[430] == 1


@pytest.mark.unit
def test_chunk_size_zero_with_train_tokenizer_disables_chunking_and_truncates(train_tokenizer):
    """
    With passage_chunk_size > 0, TrainCollator should take the chunking path.
    
    Tests chunked passages with passage_max_len=64 and passage_chunk_size=32.
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
        passage_chunk_size=32,
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
    # Since passage_chunk_size > 0 (chunking enabled), TrainCollator.__call__() returns:
    #   (q_batch, p_batch, eos_positions)  # 3-element tuple
    #
    # Where:
    #   q_batch: dict with PyTorch tensors for queries
    #     - "input_ids": tensor([[token_ids for "q1"]])  # shape: [num_queries, query_seq_len]
    #     - "attention_mask": tensor([[1, 1, ...]])      # shape: [num_queries, query_seq_len]
    #
    #   p_batch: dict with PyTorch tensors for chunked passages (FLATTENED across all queries)
    #     - "input_ids": tensor([
    #         [token_ids for passage 1 (chunked, padded to multiple of 16)],
    #         [token_ids for passage 2 (chunked, padded to multiple of 16)]
    #       ])  # shape: [total_passages, passage_seq_len]
    #     - "attention_mask": tensor([
    #         [1, 1, ..., 0, 0, ...],  # attention mask with padding
    #         [1, 1, ..., 0, 0, ...]
    #       ])  # shape: [total_passages, passage_seq_len]
    #
    #   eos_positions: list of lists, e.g., [[31, 63], [31, 63]] - EOS token positions per passage
    #                  Used by the model to extract chunk embeddings via MaxSim pooling
    #
    # Note: The collator flattens all passages from all queries into a single batch.
    #       With 1 query and train_group_size=2, we get 2 passages in p_batch.
    # ========================================================================
    out = collator(features)
    assert len(out) == 3  # Verify chunking path returns 3 elements
    q_batch, p_batch, eos_positions = out  # Unpack: q_batch (queries), p_batch (passages), eos_positions

    assert p_batch["input_ids"].shape[0] == 2  # train_group_size=2
    assert len(eos_positions) == 2  # One list of eos positions per passage

    for i in range(p_batch["input_ids"].shape[0]):
        got_ids = p_batch["input_ids"][i].tolist()
        got_mask = p_batch["attention_mask"][i].tolist()
        unpadded_len = sum(got_mask)

        # Verify chunking structure
        assert len(eos_positions[i]) > 0  # Should have at least one chunk
        assert _strictly_increasing(eos_positions[i])  # EOS positions should be strictly increasing
        
        # Verify EOS tokens at eos positions
        for eos_pos in eos_positions[i]:
            assert got_ids[eos_pos] == train_tokenizer.eos_token_id
            assert got_mask[eos_pos] == 1  # EOS tokens should be attended
        eos_positions[0][0] == 31
        eos_positions[0][1] == 63
        eos_positions[1][0] == 31
        eos_positions[1][1] == 63
        # Verify padding to multiple of 16
        assert len(got_ids) == 64
        assert len(got_mask) == 64
        assert len(got_ids) == len(got_mask)


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
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([long_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, EOS_TOKEN_ID
    ]
    expected_eos_positions = [[63]]
    expected_mask = [1] * 64

    assert sum(mask) == 64
    assert len(ids) == 64
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert ids[63] == EOS_TOKEN_ID
    assert EOS_TOKEN_ID not in ids[0:63] # EOS token should not be in the first 63 tokens
    assert mask == expected_mask
    assert _strictly_increasing(eos_positions[0])


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
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([long_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output (same as chunk_size == max_len case)
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, EOS_TOKEN_ID
    ]
    expected_eos_positions = [[63]]
    expected_mask = [1] * 64

    assert sum(mask) == 64
    assert len(ids) == 64
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert ids[63] == EOS_TOKEN_ID
    assert mask == expected_mask
    assert _strictly_increasing(eos_positions[0])


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
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([short_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: "Hello world" -> 2 tokens + 1 EOS = 3 tokens, padded to 16
    expected_ids = [9707, 1879, EOS_TOKEN_ID] + [PADDING_TOKEN_ID] * 13  # 3 content + 13 padding
    expected_eos_positions = [[2]]
    expected_mask = [1, 1, 1] + [0] * 13  # 3 ones + 13 zeros

    assert sum(mask) == 3
    assert len(ids) == 16  # Padded to multiple of 16
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert ids[2] == EOS_TOKEN_ID  # EOS at position 2
    assert mask == expected_mask
    assert _strictly_increasing(eos_positions[0])


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
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: 50 unpadded tokens (2 chunks: 31+1 EOS, 18+1 EOS), padded to 64
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        EOS_TOKEN_ID, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629,
        279, 9981, 57330, EOS_TOKEN_ID
    ] + [PADDING_TOKEN_ID] * 14  # 50 content + 14 padding
    expected_eos_positions = [[31, 49]]
    expected_mask = [1] * 50 + [0] * 14  # 50 ones + 14 zeros
    assert sum(mask) == 50
    assert len(ids) == 64  # Padded to multiple of 16
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert ids[31] == EOS_TOKEN_ID  # First EOS
    assert ids[49] == EOS_TOKEN_ID  # Second EOS
    assert mask == expected_mask
    assert _strictly_increasing(eos_positions[0])


@pytest.mark.unit
def test_chunking_multiple_passages_different_lengths(train_tokenizer):
    """
    Test batch processing with multiple passages of different lengths:
    - Short passage (2 tokens)
    - Medium passage (18 tokens)
    - Long passage (128 tokens, multiple chunks)
    - Very long passage (158 tokens, multiple chunks)
    All should be padded to the same length (longest unpadded length rounded up to pad_to_multiple_of).
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    # Create a passage that will result in ~158 tokens
    # REAL_TEXT is ~431 tokens, so we'll use a portion of it repeated or extended
    long_passage = REAL_TEXT + " " + REAL_TEXT[:200]
    
    texts = ["Short", REAL_TEXT[:100], REAL_TEXT, long_passage]
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages(texts)
    
    expected_ids_0 = [12472, EOS_TOKEN_ID] + [PADDING_TOKEN_ID] * 126
    expected_mask_0 = [1, 1] + [0] * 126
    expected_eos_0 = [1]
    
    # Passage 1: REAL_TEXT[:100] -> 17 tokens + 1 EOS = 18 tokens, padded to 160
    expected_ids_1 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        1062, EOS_TOKEN_ID
    ] + [PADDING_TOKEN_ID] * 110
    expected_mask_1 = [1] * 18 + [0] * 110
    expected_eos_1 = [17]
    
    # Passage 2: REAL_TEXT -> 2 chunks (63+1 EOS, 63+1 EOS) = 128 tokens, padded to 160
    expected_ids_2 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, EOS_TOKEN_ID, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77,
        284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239,
        315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434,
        320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622,
        EOS_TOKEN_ID
    ]
    expected_mask_2 = [1] * 128
    expected_eos_2 = [63, 127]
    
    expected_ids_3 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, EOS_TOKEN_ID, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77,
        284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239,
        315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434,
        320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622,
        EOS_TOKEN_ID
    ]
    expected_mask_3 = [1] * 128
    expected_eos_3 = [63, 127]

    ids_0 = d_collated["input_ids"][0].tolist()
    mask_0 = d_collated["attention_mask"][0].tolist()
    ids_1 = d_collated["input_ids"][1].tolist()
    mask_1 = d_collated["attention_mask"][1].tolist()
    ids_2 = d_collated["input_ids"][2].tolist()
    mask_2 = d_collated["attention_mask"][2].tolist()
    ids_3 = d_collated["input_ids"][3].tolist()
    mask_3 = d_collated["attention_mask"][3].tolist()

    # Passage 0 assertions
    assert sum(mask_0) == 2
    assert len(ids_0) == 128
    assert ids_0 == expected_ids_0
    assert mask_0 == expected_mask_0
    assert eos_positions[0] == expected_eos_0
    
    # Passage 1 assertions
    assert sum(mask_1) == 18
    assert len(ids_1) == 128
    assert ids_1 == expected_ids_1
    assert mask_1 == expected_mask_1
    assert eos_positions[1] == expected_eos_1
    
    # Passage 2 assertions
    assert sum(mask_2) == 128
    assert len(ids_2) == 128
    assert ids_2 == expected_ids_2
    assert mask_2 == expected_mask_2
    assert eos_positions[2] == expected_eos_2
    assert _strictly_increasing(eos_positions[2])
    
    # Passage 3 assertions
    assert sum(mask_3) == 128
    assert len(ids_3) == 128
    assert eos_positions[3] == expected_eos_3
    assert ids_3 == expected_ids_3
    assert mask_3 == expected_mask_3


@pytest.mark.unit
def test_non_chunked_padding_side_behavior(train_tokenizer):
    """
    Test non-chunked passage encoding behavior with left vs right padding.
    This verifies that padding_side affects how _pooling('last'/'eos') extracts embeddings.
    """
    import torch
    from unittest.mock import Mock
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator
    from tevatron.retriever.modeling.dense import DenseModel
    
    # Test passage - will be truncated to max_len
    test_passage = REAL_TEXT  # Long passage that will be truncated
    
    # Test Case 1: Right padding
    data_args_right = DataArguments(
        passage_max_len=64,
        passage_chunk_size=0,  # No chunking
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        append_eos_token=False,
    )
    
    collator_right = TrainCollator(data_args=data_args_right, tokenizer=train_tokenizer)
    q_batch_right, p_batch_right = collator_right([("query", [test_passage], [])])
    
    # Verify right padding structure
    input_ids_right = p_batch_right['input_ids'][0]
    attention_mask_right = p_batch_right['attention_mask'][0]
    seq_len_right = len(attention_mask_right)
    
    # With right padding, content tokens are at the beginning, padding at the end
    # Last position should be padding (since passage is truncated and padded)
    # Note: first position might be special token (BOS) due to add_special_tokens=True
    assert attention_mask_right[-1] == 0, "Right padding: last position should be padding"
    
    # Last valid token position
    last_valid_pos_right = attention_mask_right.sum().item() - 1
    
    # Test Case 2: Left padding
    data_args_left = DataArguments(
        passage_max_len=64,
        passage_chunk_size=0,  # No chunking
        pad_to_multiple_of=16,
        padding_side="left",
        passage_prefix="",
        append_eos_token=False,
    )
    
    collator_left = TrainCollator(data_args=data_args_left, tokenizer=train_tokenizer)
    q_batch_left, p_batch_left = collator_left([("query", [test_passage], [])])
    
    # Verify left padding structure
    input_ids_left = p_batch_left['input_ids'][0]
    attention_mask_left = p_batch_left['attention_mask'][0]
    seq_len_left = len(attention_mask_left)
    
    # With left padding, padding tokens are at the beginning, content at the end
    # Due to pad_to_multiple_of, the actual behavior depends on content length
    # Key observation: The pooling logic checks if last position is valid to determine left padding
    num_valid_left = attention_mask_left.sum().item()
    
    # The _pooling logic: left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    # If last position is 1 for all sequences, it treats it as left padding
    is_detected_as_left_padding = (attention_mask_left[-1] == 1).item()
    
    # Verify both versions tokenized the same content (ignoring padding)
    content_tokens_right = input_ids_right[attention_mask_right.bool()].tolist()
    content_tokens_left = input_ids_left[attention_mask_left.bool()].tolist()
    assert content_tokens_right == content_tokens_left, "Content tokens should be identical"
    
    # Test Case 3: Verify pooling behavior with mock model
    hidden_size = 64
    
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        # Create hidden states where each position encodes its position index
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                # Encode position j in the first dimension
                hidden_states[i, j, 0] = float(j)
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = 0  # No chunking
    
    # Test right padding pooling
    p_reps_right = model.encode_passage(p_batch_right)
    
    # Test left padding pooling
    p_reps_left = model.encode_passage(p_batch_left)
    
    # Verify pooling extracts from correct positions
    # Right padding: uses sequence_lengths calculation (attention_mask.sum() - 1)
    expected_pos_right = last_valid_pos_right
    assert torch.allclose(p_reps_right[0, 0], torch.tensor(float(expected_pos_right)))
    
    # Left padding: The _pooling logic checks: left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    # If last position is 1, it uses last_hidden_state[:, -1]
    # Otherwise, it calculates sequence_lengths = attention_mask.sum(dim=1) - 1
    if is_detected_as_left_padding:
        expected_pos_left = seq_len_left - 1
    else:
        expected_pos_left = num_valid_left - 1
    assert torch.allclose(p_reps_left[0, 0], torch.tensor(float(expected_pos_left)))
    
    # Verify the key difference: right padding always uses sequence_lengths calculation
    # Left padding uses last position if detected as left padding, otherwise sequence_lengths
    # The actual positions depend on the padding structure
    print(f"Right padding: extracted from position {expected_pos_right} (last_valid_pos)")
    print(f"Left padding: extracted from position {expected_pos_left} (is_left_padding={is_detected_as_left_padding})")
    print(f"Right padding mask: first={attention_mask_right[0].item()}, last={attention_mask_right[-1].item()}")
    print(f"Left padding mask: first={attention_mask_left[0].item()}, last={attention_mask_left[-1].item()}")


@pytest.mark.unit
def test_chunked_passages_left_padding(train_tokenizer):
    """
    Test chunked passage encoding with left padding.
    This verifies that EOS positions are correctly adjusted when padding is on the left.
    """
    import torch
    from unittest.mock import Mock
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator
    from tevatron.retriever.modeling.dense import DenseModel
    
    # Test passage that will be chunked
    test_passage = REAL_TEXT
    
    # Test Case 1: Right padding (baseline)
    data_args_right = DataArguments(
        passage_max_len=128,
        passage_chunk_size=64,
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        append_eos_token=False,
    )
    
    collator_right = TrainCollator(data_args=data_args_right, tokenizer=train_tokenizer)
    q_batch_right, p_batch_right, eos_positions_right = collator_right([("query", [test_passage], [])])
    
    # Verify right padding structure
    input_ids_right = p_batch_right['input_ids'][0]
    attention_mask_right = p_batch_right['attention_mask'][0]
    seq_len_right = len(attention_mask_right)
    
    # With right padding, content tokens are at the beginning, padding at the end
    assert attention_mask_right[-1] == 0, "Right padding: last position should be padding"
    
    # Verify EOS positions are correct (should be in the content area, before padding)
    for eos_pos in eos_positions_right[0]:
        assert eos_pos < attention_mask_right.sum().item(), f"EOS position {eos_pos} should be in valid token range"
        assert input_ids_right[eos_pos] == train_tokenizer.eos_token_id, f"Position {eos_pos} should be EOS token"
    
    # Test Case 2: Left padding
    data_args_left = DataArguments(
        passage_max_len=128,
        passage_chunk_size=64,
        pad_to_multiple_of=16,
        padding_side="left",
        passage_prefix="",
        append_eos_token=False,
    )
    
    collator_left = TrainCollator(data_args=data_args_left, tokenizer=train_tokenizer)
    q_batch_left, p_batch_left, eos_positions_left = collator_left([("query", [test_passage], [])])
    
    # Verify left padding structure
    input_ids_left = p_batch_left['input_ids'][0]
    attention_mask_left = p_batch_left['attention_mask'][0]
    seq_len_left = len(attention_mask_left)
    
    # With left padding, padding tokens are at the beginning, content at the end
    # Note: Due to pad_to_multiple_of, the actual padding structure may vary
    # Check that there is padding at the beginning
    num_valid_tokens = attention_mask_left.sum().item()
    padding_length = seq_len_left - num_valid_tokens
    if padding_length > 0:
        # If there's padding, first positions should be padding
        assert attention_mask_left[0] == 0, "Left padding: first position should be padding when padding exists"
    assert attention_mask_left[-1] == 1, "Left padding: last position should be content (valid token)"
    
    # Verify EOS positions are correctly adjusted for left padding
    # EOS positions should be shifted by the padding length
    
    # Verify all EOS positions are in the valid token range (after padding)
    for eos_pos in eos_positions_left[0]:
        assert eos_pos >= padding_length, f"EOS position {eos_pos} should be after padding (padding_length={padding_length})"
        assert eos_pos < seq_len_left, f"EOS position {eos_pos} should be within sequence length {seq_len_left}"
        assert input_ids_left[eos_pos] == train_tokenizer.eos_token_id, f"Position {eos_pos} should be EOS token"
        assert attention_mask_left[eos_pos] == 1, f"EOS position {eos_pos} should be in valid token range"
    
    # Verify that EOS positions are correctly shifted
    # The relative positions within the content should be the same, but absolute positions differ
    # Right padding: EOS at positions like [63, 127] (before padding)
    # Left padding: EOS at positions like [padding_length + 63, padding_length + 127] (after padding)
    assert len(eos_positions_right[0]) == len(eos_positions_left[0]), "Should have same number of chunks"
    
    # Verify the relative positions are preserved (EOS positions differ by padding_length)
    for i, (eos_right, eos_left) in enumerate(zip(eos_positions_right[0], eos_positions_left[0])):
        expected_left_pos = eos_right + padding_length
        assert eos_left == expected_left_pos, \
            f"Chunk {i}: EOS position should be shifted by padding_length. " \
            f"Expected {expected_left_pos}, got {eos_left} (right={eos_right}, padding_length={padding_length})"
    
    # Test Case 3: Verify pooling behavior with mock model
    hidden_size = 64
    
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        # Create hidden states where each position encodes its position index
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                # Encode position j in the first dimension
                hidden_states[i, j, 0] = float(j)
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = 64
    
    # Test right padding pooling
    chunk_reps_right, chunk_mask_right = model.encode_passage(p_batch_right, eos_positions_right)
    
    # Test left padding pooling
    chunk_reps_left, chunk_mask_left = model.encode_passage(p_batch_left, eos_positions_left)
    
    # Verify pooling extracts from correct EOS positions
    # Right padding: extracts from eos_positions_right
    # Left padding: extracts from eos_positions_left (which are adjusted)
    assert chunk_reps_right.shape == chunk_reps_left.shape, "Should have same number of chunks"
    assert chunk_mask_right.shape == chunk_mask_left.shape, "Should have same chunk mask shape"
    
    # Verify that embeddings are extracted from the correct positions
    # For right padding, EOS at position 63 should give embedding with value 63.0
    # For left padding, EOS at position (padding_length + 63) should give embedding with value (padding_length + 63.0)
    for i, (eos_right, eos_left) in enumerate(zip(eos_positions_right[0], eos_positions_left[0])):
        # Right padding: embedding should encode position eos_right
        assert torch.allclose(chunk_reps_right[0, i, 0], torch.tensor(float(eos_right))), \
            f"Right padding chunk {i}: embedding should encode EOS position {eos_right}"
        
        # Left padding: embedding should encode position eos_left
        assert torch.allclose(chunk_reps_left[0, i, 0], torch.tensor(float(eos_left))), \
            f"Left padding chunk {i}: embedding should encode EOS position {eos_left}"
        
        # Verify masks are correct
        assert chunk_mask_right[0, i] == 1.0, f"Right padding chunk {i} should be valid"
        assert chunk_mask_left[0, i] == 1.0, f"Left padding chunk {i} should be valid"
    
    # Verify that the embeddings differ by the padding length (in the first dimension)
    # This confirms that EOS positions are correctly adjusted
    for i in range(len(eos_positions_right[0])):
        expected_diff = float(padding_length)
        actual_diff = chunk_reps_left[0, i, 0] - chunk_reps_right[0, i, 0]
        assert torch.allclose(actual_diff, torch.tensor(expected_diff)), \
            f"Chunk {i}: embedding difference should equal padding_length. " \
            f"Expected {expected_diff}, got {actual_diff.item()}"
    
    print(f"Right padding EOS positions: {eos_positions_right[0]}")
    print(f"Left padding EOS positions: {eos_positions_left[0]}")
    print(f"Padding length: {padding_length}")
    print(f"Sequence length: {seq_len_left}")
    print(f"Valid tokens: {num_valid_tokens}")
    
    # Test Case 4: Verify with append_eos_token=True
    data_args_right_eos = DataArguments(
        passage_max_len=64,
        passage_chunk_size=0,
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        append_eos_token=True,
    )
    
    data_args_left_eos = DataArguments(
        passage_max_len=64,
        passage_chunk_size=0,
        pad_to_multiple_of=16,
        padding_side="left",
        passage_prefix="",
        append_eos_token=True,
    )
    
    collator_right_eos = TrainCollator(data_args=data_args_right_eos, tokenizer=train_tokenizer)
    collator_left_eos = TrainCollator(data_args=data_args_left_eos, tokenizer=train_tokenizer)
    
    q_batch_eos_right, p_batch_eos_right = collator_right_eos([("query", [test_passage], [])])
    q_batch_eos_left, p_batch_eos_left = collator_left_eos([("query", [test_passage], [])])
    
    # Verify EOS token is present in both
    content_right_eos = p_batch_eos_right['input_ids'][0][p_batch_eos_right['attention_mask'][0].bool()].tolist()
    content_left_eos = p_batch_eos_left['input_ids'][0][p_batch_eos_left['attention_mask'][0].bool()].tolist()
    
    assert content_right_eos[-1] == train_tokenizer.eos_token_id
    assert content_left_eos[-1] == train_tokenizer.eos_token_id
    
    # Test pooling with EOS
    p_reps_eos_right = model.encode_passage(p_batch_eos_right)
    p_reps_eos_left = model.encode_passage(p_batch_eos_left)
    
    # Both should extract from EOS position
    mask_eos_right = p_batch_eos_right['attention_mask'][0]
    mask_eos_left = p_batch_eos_left['attention_mask'][0]
    
    # Right padding: uses sequence_lengths calculation
    last_valid_eos_right = mask_eos_right.sum().item() - 1
    
    # Left padding: checks if last position is valid
    is_left_padding_eos = (mask_eos_left[-1] == 1).item()
    if is_left_padding_eos:
        last_valid_eos_left = mask_eos_left.shape[0] - 1
    else:
        last_valid_eos_left = mask_eos_left.sum().item() - 1
    
    assert torch.allclose(p_reps_eos_right[0, 0], torch.tensor(float(last_valid_eos_right)))
    assert torch.allclose(p_reps_eos_left[0, 0], torch.tensor(float(last_valid_eos_left)))
    
    # With EOS, the extracted positions should be where EOS is located
    assert p_batch_eos_right['input_ids'][0][last_valid_eos_right] == train_tokenizer.eos_token_id
    assert p_batch_eos_left['input_ids'][0][last_valid_eos_left] == train_tokenizer.eos_token_id
    
    # Summary: This test verifies that padding_side affects pooling position calculation
    # Right padding: always uses attention_mask.sum() - 1
    # Left padding: uses seq_len - 1 if last position is valid, otherwise attention_mask.sum() - 1
