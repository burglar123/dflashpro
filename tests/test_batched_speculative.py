from types import SimpleNamespace

import torch
from transformers import DynamicCache

from benchmark import (
    BlockManager,
    Scheduler,
    Sequence,
    dflash_generate,
    dflash_generate_batch,
    gather_active_rows,
    init_batch_state,
    prepare_packed_verify_inputs,
    scatter_back_rows,
    sync_sequence_blocks,
    target_verify_step,
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


def test_block_manager_prefix_cache_reuses_blocks_and_tracks_refcount():
    manager = BlockManager(block_size=2)
    seq_a = Sequence(
        seq_id=0,
        token_ids=[1, 2, 3, 4, 5],
        num_cached_tokens=4,
        block_table=[],
        pre_verify=True,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )
    seq_b = Sequence(
        seq_id=1,
        token_ids=[1, 2, 3, 9, 10],
        num_cached_tokens=4,
        block_table=[],
        pre_verify=True,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )

    sync_sequence_blocks(seq_a, manager)
    sync_sequence_blocks(seq_b, manager)

    assert seq_a.block_table[0].physical_block_id == seq_b.block_table[0].physical_block_id
    assert seq_a.block_table[1].physical_block_id != seq_b.block_table[1].physical_block_id


def test_block_manager_rollback_and_consistency():
    manager = BlockManager(block_size=3)
    seq = Sequence(
        seq_id=0,
        token_ids=[7, 8, 9, 10, 11, 12],
        num_cached_tokens=6,
        block_table=[],
        pre_verify=False,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )

    sync_sequence_blocks(seq, manager)
    seq.num_cached_tokens = 4
    manager.rollback_sequence(seq, seq.num_cached_tokens)
    sync_sequence_blocks(seq, manager)

    assert len(seq.block_table) == 2
    assert seq.block_table[1].logical_start == 3
    assert seq.block_table[1].logical_end == 4


def test_prepare_packed_verify_inputs_respects_pre_verify_lengths():
    seqs = [
        Sequence(
            seq_id=0,
            token_ids=[1, 2, 3],
            num_cached_tokens=2,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[4, 5, 6, 7],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]
    packed = prepare_packed_verify_inputs(seqs, gamma=4)
    assert packed["verify_lens"] == [1, 4]
    assert packed["packed_ranges"] == [(0, 1), (1, 5)]
    assert packed["slot_mapping"] == [0, 1, 1, 1, 1]


def test_target_verify_step_handles_ragged_verify_lengths_without_misalignment():
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    state, _ = init_batch_state(
        model=model,
        target=target,
        input_ids=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        input_lengths=torch.tensor([3, 3], dtype=torch.long),
        mask_token_id=model.mask_token_id,
        max_new_tokens=4,
        block_size=3,
        temperature=0.0,
    )

    scheduled_batch = [
        Sequence(
            seq_id=0,
            token_ids=[10, 11, 12, 13],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[20, 21, 22, 23],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]

    active_indices = torch.tensor([0, 1], dtype=torch.long)
    block_output_ids = torch.tensor([[13, 99, 99], [23, 24, 25]], dtype=torch.long)
    block_position_ids = torch.tensor([[3, 4, 5], [3, 4, 5]], dtype=torch.long)

    posterior, acceptance_len_vec, _, _ = target_verify_step(
        model=model,
        target=target,
        state=state,
        scheduled_batch=scheduled_batch,
        active_indices=active_indices,
        block_output_ids=block_output_ids,
        block_position_ids=block_position_ids,
        block_size=3,
        temperature=0.0,
    )

    assert acceptance_len_vec.tolist() == [0, 2]
    assert posterior[0, 0].item() == 14
    assert posterior[1, 0].item() == 24


def _build_sequence(seq_id: int, token_ids: list[int], num_cached_tokens: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        token_ids=list(token_ids),
        num_cached_tokens=num_cached_tokens,
        block_table=[],
        pre_verify=False,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )


def test_scheduler_and_block_manager_rollback_are_atomic():
    manager = BlockManager(block_size=2)
    seq = _build_sequence(seq_id=0, token_ids=[10, 11], num_cached_tokens=2)
    manager.register_sequence(seq)
    sync_sequence_blocks(seq, manager)

    scheduler = Scheduler([seq], block_manager=manager)
    scheduler.schedule_next_batch()
    scheduler.append_draft_tokens(seq_id=0, draft_tokens=[12, 13, 14])

    scheduler.rollback(seq_id=0, rollback_to=3)

    assert seq.num_cached_tokens == 3
    assert seq.token_ids == [10, 11, 12]
    assert all(mapping.logical_start < 3 for mapping in seq.block_table)
    manager.check_consistency(seq)


def test_transactional_accept_reject_paths_cover_all_reject_mid_reject_all_accept():
    scenarios = [
        ("all_reject", 0, [20, 21, 99]),
        ("mid_reject", 2, [20, 21, 30, 31, 99]),
        ("all_accept", 4, [20, 21, 30, 31, 32, 33, 99]),
    ]

    for _, accepted_draft_len, expected_tokens in scenarios:
        manager = BlockManager(block_size=2)
        seq = _build_sequence(seq_id=0, token_ids=[20, 21], num_cached_tokens=2)
        manager.register_sequence(seq)
        sync_sequence_blocks(seq, manager)
        scheduler = Scheduler([seq], block_manager=manager)
        scheduler.schedule_next_batch()

        draft_tokens = [30, 31, 32, 33]
        scheduler.append_draft_tokens(seq_id=0, draft_tokens=draft_tokens)
        txn = scheduler.consume_draft_transaction(seq_id=0)
        draft_start = int(txn["start"])

        scheduler.rollback(seq_id=0, rollback_to=draft_start + accepted_draft_len)
        seq.token_ids = seq.token_ids[: seq.num_cached_tokens] + [99]
        seq.num_cached_tokens += 1
        sync_sequence_blocks(seq, manager)

        assert seq.token_ids == expected_tokens
        assert seq.num_cached_tokens == len(expected_tokens)
        manager.check_consistency(seq)


def test_eos_hit_removes_sequence_from_running_and_releases_recyclable_blocks():
    manager = BlockManager(block_size=2)
    seq0 = _build_sequence(seq_id=0, token_ids=[1, 2], num_cached_tokens=2)
    seq1 = _build_sequence(seq_id=1, token_ids=[3, 4], num_cached_tokens=2)
    for seq in (seq0, seq1):
        manager.register_sequence(seq)
        sync_sequence_blocks(seq, manager)

    scheduler = Scheduler([seq0, seq1], block_manager=manager)
    scheduler.schedule_next_batch()

    scheduler.append_draft_tokens(seq_id=0, draft_tokens=[5, 6, 7, 8])
    free_before = len(manager._free_block_ids)
    scheduler.rollback(seq_id=0, rollback_to=2)
    free_after = len(manager._free_block_ids)

    scheduler.mark_finished(seq0)

    assert seq0.finished is True
    assert seq0 not in scheduler.running
    assert seq0 in scheduler.finished
    assert free_after >= free_before + 2
