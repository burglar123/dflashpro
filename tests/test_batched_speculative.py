from types import SimpleNamespace

import torch
from transformers import DynamicCache

from benchmark import (
    dflash_generate,
    dflash_generate_batch,
    gather_active_rows,
    scatter_back_rows,
)


class DummyDraftModel:
    def __init__(self, device: torch.device):
        self.device = device
        self.mask_token_id = -1
        self.target_layer_ids = [0]


class DummyTargetModel:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(embed_tokens=lambda token_ids: torch.zeros((*token_ids.shape, 8)))

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        return torch.zeros((bsz, seq_len, self.vocab_size), dtype=torch.float32)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=None,
        output_hidden_states=False,
    ):
        bsz, seq_len = input_ids.shape
        logits = torch.full((bsz, seq_len, self.vocab_size), -1e9, dtype=torch.float32)
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)

        hidden = torch.zeros((bsz, seq_len, 8), dtype=torch.float32)
        hidden_states = [hidden, hidden] if output_hidden_states else None

        if past_key_values is not None:
            if not hasattr(past_key_values, "key_cache"):
                past_key_values.key_cache = []
                past_key_values.value_cache = []
            if not past_key_values.key_cache:
                past_key_values.key_cache = [torch.zeros((bsz, 1, seq_len, 4), dtype=torch.float32)]
                past_key_values.value_cache = [torch.zeros((bsz, 1, seq_len, 4), dtype=torch.float32)]

        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


def run_single(model, target, prompt_ids, max_new_tokens=8):
    with torch.inference_mode():
        out = dflash_generate(
            model=model,
            target=target,
            input_ids=prompt_ids,
            mask_token_id=model.mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=1,
            stop_token_ids=[255],
            temperature=0.0,
        )[0]
    return out.output_ids[0]


def test_batch1_matches_single_path_temperature_zero():
    torch.manual_seed(0)
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)

    solo_ids = run_single(model, target, prompt)

    with torch.inference_mode():
        batch_out = dflash_generate_batch(
            model=model,
            target=target,
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            input_lengths=torch.tensor([prompt.shape[1]], dtype=torch.long),
            mask_token_id=model.mask_token_id,
            max_new_tokens=8,
            block_size=1,
            stop_token_ids=[255],
            temperature=0.0,
        )[0]

    assert torch.equal(solo_ids, batch_out.output_ids[0])


def test_batch2_and_batch4_no_cross_contamination():
    torch.manual_seed(0)
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    prompts = torch.tensor(
        [
            [1, 2, 3],
            [10, 11, 12],
            [42, 43, 44],
            [100, 101, 102],
        ],
        dtype=torch.long,
    )

    for batch_size in (2, 4):
        batch_prompts = prompts[:batch_size]
        with torch.inference_mode():
            batch_responses = dflash_generate_batch(
                model=model,
                target=target,
                input_ids=batch_prompts,
                attention_mask=torch.ones_like(batch_prompts),
                input_lengths=torch.full((batch_size,), batch_prompts.shape[1], dtype=torch.long),
                mask_token_id=model.mask_token_id,
                max_new_tokens=6,
                block_size=1,
                stop_token_ids=[255],
                temperature=0.0,
            )

        for row in range(batch_size):
            row_tokens = batch_responses[row].output_ids[0]
            assert (row_tokens >= 0).all(), "generated row must not contain mask token"

            solo = run_single(model, target, batch_prompts[row : row + 1], max_new_tokens=6)
            assert torch.equal(row_tokens, solo), f"row {row} changed under batch={batch_size}"


def test_stage_modes_are_rollback_safe():
    torch.manual_seed(0)
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)
    prompt = torch.tensor(
        [
            [3, 4, 5],
            [30, 31, 32],
        ],
        dtype=torch.long,
    )

    for mode in [
        "legacy",
        "stage_a_prefill_only",
        "stage_b_target_only",
        "stage_c_full_speculative",
    ]:
        with torch.inference_mode():
            responses = dflash_generate_batch(
                model=model,
                target=target,
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                input_lengths=torch.full((prompt.shape[0],), prompt.shape[1], dtype=torch.long),
                mask_token_id=model.mask_token_id,
                max_new_tokens=5,
                block_size=1,
                stop_token_ids=[255],
                temperature=0.0,
                batched_decode_mode=mode,
            )
        assert len(responses) == prompt.shape[0]
        for row in range(prompt.shape[0]):
            row_tokens = responses[row].output_ids[0]
            assert torch.equal(row_tokens[: prompt.shape[1]], prompt[row])
            assert (row_tokens >= 0).all()


def test_cache_alignment_after_active_row_compaction_and_scatter():
    full_cache = DynamicCache()
    full_cache.key_cache = [
        torch.arange(5 * 2 * 4 * 3, dtype=torch.float32).reshape(5, 2, 4, 3)
    ]
    full_cache.value_cache = [
        torch.arange(5 * 2 * 4 * 3, dtype=torch.float32).reshape(5, 2, 4, 3) + 10_000
    ]

    active_indices = torch.tensor([1, 3], dtype=torch.long)
    active_cache = gather_active_rows(full_cache, active_indices)

    # Simulate per-token updates in compacted active cache.
    active_cache.key_cache[0][:, :, 2:, :] = -7
    active_cache.value_cache[0][:, :, 1:3, :] = -9

    original_inactive_key = full_cache.key_cache[0][0].clone()
    original_inactive_value = full_cache.value_cache[0][4].clone()

    scatter_back_rows(full_cache, active_cache, active_indices)

    assert torch.equal(full_cache.key_cache[0][0], original_inactive_key)
    assert torch.equal(full_cache.value_cache[0][4], original_inactive_value)

    assert torch.equal(full_cache.key_cache[0][1, :, 2:, :], torch.full((2, 2, 3), -7.0))
    assert torch.equal(full_cache.key_cache[0][3, :, 2:, :], torch.full((2, 2, 3), -7.0))
    assert torch.equal(full_cache.value_cache[0][1, :, 1:3, :], torch.full((2, 2, 3), -9.0))
    assert torch.equal(full_cache.value_cache[0][3, :, 1:3, :], torch.full((2, 2, 3), -9.0))
