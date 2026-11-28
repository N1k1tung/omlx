# SPDX-License-Identifier: Apache-2.0
"""Block drafting for several requests whose verify blocks share one forward.

mlx-vlm's block-draft round loop serves one request, and a second request used
to decode without drafting. Its batched form merges the requests' caches into
one left-padded batch; a round in which the requests accept different numbers
of tokens then shifts whole cache rows, which costs more the longer the
context is.

Here every request keeps its own cache and its own drafter state, and only the
forwards are shared: ``speculative_verify_rows`` lays the requests' blocks end
to end and reads the target's weights once for all the rows, and
``draft_block_bases`` does the same for the drafter. The blocks may differ in
length, nothing is padded, and each request commits or rolls back on its own.
A round with one request is the stock round.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx_vlm.speculative import common as _vlm_common
from mlx_vlm.speculative import dflash as _vlm_dflash
from mlx_vlm.speculative.cache_state import (
    abort_speculative_round,
    commit_speculative_round,
)
from mlx_vlm.speculative.common import (  # noqa: SLF001
    _copy_rng_state,
    _dflash_block_total,
    _record_speculative_round,
    _restore_rng_state,
    _speculative_walk,
    _SpeculativeSamplerRNG,
)
from mlx_vlm.speculative.dflash import (  # noqa: SLF001
    _PositionedDraftSampler,
    _reserve_dflash_target_cache,
    _sample_dflash_target_walk,
    _supports_positioned_target_sampling,
)

from ..utils.metal_sync import _sync_and_clear_cache

# Requests that draft together. The verify forward holds eight rows and every
# request needs its anchor and one proposal. Measured on ERNIE 5.5 Flash,
# total tok/s: four requests of two rows each give 132 on prose and 141 on
# code (122 and 128 sampled) against 147 for four rows of plain batched
# decode. A verify row past the fourth costs about 3 ms, as much as a row of
# plain decode, and a drafted token is only sometimes kept.
MAX_REQUESTS = 3
# Requests that begin drafting together when none drafts yet. Measured on
# ERNIE 5.5 Flash: two draft together faster than they batch without
# drafting (110 and 123 against 85). Three do not: 124 on prose and 129 to
# 135 on code against 125, and 113 and 119 sampled. A third still joins
# requests that already draft, since those cannot go back.
START_TOGETHER = 2
# Reach probability a proposal needs when this many requests share a round.
# One request keeps the drafter's own thresholds. With more, the forward's
# fixed cost is already spread over the requests, so a row has to earn more.
DEPTH_FLOOR = {2: 0.30, 3: 0.38}
# Rounds between buffer-pool drains (see vlm_mtp._BLOCK_DRAFT_CLEAR_ROUNDS).
_CLEAR_ROUNDS = 64


class _Row:
    """One request: its cache, its drafter state and what it has yet to emit."""

    def __init__(self) -> None:
        self.cache: list[Any] = []
        self.hidden: Any = None
        self.bonus = 0
        self.emitted = 1  # the first token is sampled before the rounds start
        self.max_tokens = 0
        self.sampler: Callable[[mx.array], mx.array] = None  # type: ignore[assignment]
        self.greedy = True
        self.positioned = False
        self.sampler_rng: Any = None
        self.rng: list[mx.array] | None = None
        self.draft_cache: Any = None
        self.draft_state: dict = {}
        self.pending: list[int] | None = None
        self.closed = False
        self.block = 0
        self.draft_tokens: Any = None

    # The scheduler reads these for its per-request log line.
    @property
    def accept_lens(self) -> list[int]:
        return self.draft_state.get("accept_lens") or []

    @property
    def draft_lens(self) -> list[int]:
        return self.draft_state.get("draft_lens") or []


class RowStream:
    """What the scheduler iterates: one list of tokens per round."""

    def __init__(self, rounds: SharedBlockRounds, row: _Row, first: int) -> None:
        self._rounds = rounds
        self.row = row
        self._first: int | None = first

    def __iter__(self) -> RowStream:
        return self

    def __next__(self) -> list[int]:
        row = self.row
        if row.closed:
            raise StopIteration
        if self._first is not None:
            # A round another request started may already hold this one's
            # first tokens; emit them with the first token.
            tokens = [self._first] + (row.pending or [])
            self._first, row.pending = None, None
            return tokens
        tokens = self._rounds.take(row)
        if not tokens:
            self.close()
            raise StopIteration
        return tokens

    def close(self) -> None:
        self._rounds.remove(self.row)


class SharedBlockRounds:
    """Draft and verify rounds for the requests currently drafting."""

    def __init__(
        self,
        target: Any,
        draft: Any,
        *,
        block_size: int | None = None,
        token_dtype: mx.Dtype = mx.int32,
        max_requests: int = MAX_REQUESTS,
    ) -> None:
        self.target = target
        # What mlx-vlm's own loop verifies through: the language model, or
        # the adapter proxy that stands in for it.
        self.lm = getattr(target, "language_model", target)
        self.draft = draft
        self.block_size = block_size
        self.block_total = _dflash_block_total(draft, block_size)
        # The model object the scheduler routes with (set by the caller).
        self.adapter: Any = None
        self.limit = int(getattr(self.lm, "verify_rows_limit", 8))
        self.max_requests = max(1, min(int(max_requests), self.limit // 2))
        self.start_together = min(START_TOGETHER, self.max_requests)
        self.token_dtype = token_dtype
        self.rows: list[_Row] = []
        self.rounds = 0
        # Rounds whose requests went through the drafter in one forward.
        self.shared_drafts = 0
        self._cleared_at = 0

    @staticmethod
    def supported(target: Any, draft: Any) -> bool:
        """Whether ``target`` verifies several requests' blocks in one forward
        and ``draft`` can hold more than one request's state."""
        target = getattr(target, "language_model", target)
        return (
            callable(getattr(target, "speculative_verify_rows", None))
            and callable(getattr(draft, "request_state", None))
            and callable(getattr(draft, "use_request", None))
            and not callable(getattr(draft, "prepare_target_hidden", None))
        )

    def has_room(self) -> bool:
        return len(self.rows) < self.max_requests

    # -- membership ----------------------------------------------------------

    def add(
        self,
        prompt_cache: list[Any],
        hidden: mx.array,
        first_bonus: int,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        *,
        greedy: bool,
        history: list[int] | None = None,
    ) -> RowStream:
        """Admit a request after its prefill; returns its token stream."""
        draft = self.draft
        row = _Row()
        row.cache = prompt_cache
        row.hidden = hidden
        row.bonus = int(first_bonus)
        row.max_tokens = int(max_tokens)
        row.sampler = sampler
        row.greedy = bool(greedy)
        row.positioned = _supports_positioned_target_sampling(sampler)
        row.draft_cache = draft.reset(self.target)
        begin = getattr(draft, "begin_request", None)
        if callable(begin):
            begin(history)
        observe = getattr(draft, "observe", None)
        if callable(observe):
            observe(row.bonus)
        row.draft_state = draft.request_state()
        with mx.stream(_vlm_common.generation_stream):
            _reserve_dflash_target_cache(prompt_cache, self.block_total)
        row.sampler_rng = _SpeculativeSamplerRNG(
            draft, enabled=not row.greedy and not row.positioned
        )
        if row.sampler_rng.enabled:
            # Each request draws from its own stream, so its draft and its
            # verify see the same numbers whatever the other requests draw.
            row.rng = _copy_rng_state()
            # Move the shared stream on, so the next request starts elsewhere.
            mx.random.uniform(shape=(1,))
        self.rows.append(row)
        return RowStream(self, row, row.bonus)

    def remove(self, row: _Row) -> None:
        row.closed = True
        row.pending = None
        if row in self.rows:
            self.rows.remove(row)

    def row_for(self, prompt_cache: list[Any]) -> _Row | None:
        """The request that decodes into ``prompt_cache``, while it is here."""
        return next((row for row in self.rows if row.cache is prompt_cache), None)

    # -- rounds --------------------------------------------------------------

    def take(self, row: _Row) -> list[int]:
        """The request's next tokens; runs a round when it has none waiting."""
        if row.pending is None:
            self._round()
        tokens, row.pending = row.pending, None
        return tokens or []

    def _use(self, row: _Row, depth_floor: float | None = None) -> None:
        self.draft.use_request(row.draft_state, depth_floor)

    def _keep(self, row: _Row) -> None:
        row.draft_state = self.draft.request_state()

    def _next_block(
        self, row: _Row, ceiling: int, depth_floor: float | None = None
    ) -> int:
        self._use(row, depth_floor)
        size = _vlm_dflash._dflash_next_block_size(
            self.draft, ceiling, row.max_tokens - row.emitted + 1, None
        )
        self._keep(row)
        return max(1, int(size))

    def _reach(self, row: _Row, proposals: int) -> float:
        """How likely the request is to accept ``proposals`` drafts in a row."""
        reach = getattr(self.draft, "reach", None)
        if not callable(reach):
            return 1.0 / (1 + proposals)
        self._use(row)
        return float(reach(proposals))

    def _block_sizes(self, rows: list[_Row]) -> list[int]:
        """Rows each request verifies this round, within the forward's limit."""
        floor = None
        if len(rows) > 1:
            floor = DEPTH_FLOOR.get(len(rows), max(DEPTH_FLOOR.values()))
        wants = [self._next_block(row, self.block_total, floor) for row in rows]
        if sum(wants) <= self.limit:
            return wants
        # Everyone keeps an anchor and one proposal; each row that is left
        # goes to the request most likely to reach it.
        sizes = [min(want, 2) for want in wants]
        spare = self.limit - sum(sizes)
        while spare > 0:
            short = [i for i, want in enumerate(wants) if sizes[i] < want]
            if not short:
                break
            index = max(short, key=lambda i: self._reach(rows[i], sizes[i]))
            sizes[index] += 1
            spare -= 1
        # Tell the drafter the block it actually gets.
        return [
            size if size == want else self._next_block(row, size, floor)
            for row, size, want in zip(rows, sizes, wants)
        ]

    def _draft_bases(self, rows: list[_Row], sizes: list[int]) -> list[Any]:
        """One drafter forward for the requests of a round, where it can be.

        Returns each request's share of it, or ``None`` for a request that
        runs the drafter by itself: a lone request, and one whose first round
        still has its whole prompt to read into the drafter's context.
        """
        bases: list[Any] = [None] * len(rows)
        forward = getattr(self.draft, "draft_block_bases", None)
        share = [
            index
            for index, (row, size) in enumerate(zip(rows, sizes))
            if size > 1 and row.hidden.shape[1] <= self.limit
        ]
        if not callable(forward) or len(share) < 2:
            return bases
        shared = forward(
            [rows[i].bonus for i in share],
            [rows[i].hidden for i in share],
            [rows[i].draft_cache for i in share],
            [sizes[i] for i in share],
        )
        for index, base in zip(share, shared):
            bases[index] = base
        self.shared_drafts += 1
        return bases

    def _draft(self, row: _Row, size: int, base: Any = None) -> None:
        draft = self.draft
        self._use(row)
        draft_fn = (
            getattr(draft, "draft_block_greedy", draft.draft_block)
            if row.greedy
            else draft.draft_block
        )
        sampler = (
            _PositionedDraftSampler(row.sampler, row_ids=[0], positions=[row.emitted])
            if not row.greedy and row.positioned
            else row.sampler
        )
        if row.rng is not None:
            _restore_rng_state(row.rng)
        row.draft_tokens = row.sampler_rng.draft_tokens(
            draft_fn,
            row.bonus,
            row.hidden,
            row.draft_cache,
            size,
            sampler,
            self.token_dtype,
            **({} if base is None else {"base": base}),
        )
        row.block = size
        self._keep(row)

    def _round(self) -> None:
        rows = [
            row
            for row in self.rows
            if row.pending is None and not row.closed and row.emitted < row.max_tokens
        ]
        if not rows:
            return
        for start in range(0, len(rows), self.limit // 2):
            self._verify(rows[start : start + self.limit // 2])
        self.rounds += 1
        if self.rounds - self._cleared_at >= _CLEAR_ROUNDS:
            _sync_and_clear_cache(_vlm_common.generation_stream)
            self._cleared_at = self.rounds

    def _verify(self, rows: list[_Row]) -> None:
        draft, target = self.draft, self.lm
        sizes = self._block_sizes(rows)
        bases = self._draft_bases(rows, sizes)
        for row, size, base in zip(rows, sizes, bases):
            self._draft(row, size, base)

        results: list = []
        try:
            with mx.stream(_vlm_common.generation_stream):
                inputs = [
                    mx.concatenate(
                        [
                            mx.array([[row.bonus]], dtype=self.token_dtype),
                            row.draft_tokens,
                        ],
                        axis=1,
                    )
                    for row in rows
                ]
                results = target.speculative_verify_rows(
                    inputs,
                    [row.cache for row in rows],
                    capture_layer_ids=list(draft.config.target_layer_ids),
                )
            hiddens = [mx.concatenate(captured, axis=-1) for _, captured, _ in results]
            targets = [
                row.sampler(logits) if row.greedy else None
                for row, (logits, _, _) in zip(rows, results)
            ]
            mx.async_eval(*hiddens, *[t for t in targets if t is not None])

            for row, (logits, _, transaction), hidden, target_tokens in zip(
                rows, results, hiddens, targets
            ):
                budget = row.max_tokens - row.emitted
                if row.greedy:
                    accepted, tokens = _speculative_walk(
                        row.draft_tokens, target_tokens, budget
                    )
                else:
                    if row.rng is not None:
                        _restore_rng_state(row.rng)
                    accepted_list, tokens_list = _sample_dflash_target_walk(
                        logits,
                        row.draft_tokens,
                        row.sampler,
                        [budget],
                        row_ids=[0],
                        base_positions=[row.emitted],
                    )
                    accepted, tokens = accepted_list[0], tokens_list[0]
                    row.sampler_rng.target_sampled(sync_draft=not row.positioned)
                    if row.rng is not None:
                        row.rng = _copy_rng_state()
                # A clipped last round must not commit past what is emitted.
                accepted = min(int(accepted), max(len(tokens) - 1, 0))
                self._use(row)
                _record_speculative_round(draft, accepted, row.block - 1)
                observe = getattr(draft, "observe", None)
                if callable(observe):
                    for token in tokens:
                        observe(token)
                self._keep(row)
                row.hidden = hidden[:, : accepted + 1, :]
                if tokens:
                    row.bonus = tokens[-1]
                with mx.stream(_vlm_common.generation_stream):
                    commit_speculative_round(
                        target, row.cache, transaction, accepted, row.block
                    )
                row.emitted += len(tokens)
                row.pending = list(tokens)
        except BaseException:
            for _, _, transaction in results:
                abort_speculative_round(transaction)
            raise
