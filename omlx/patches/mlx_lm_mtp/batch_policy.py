# SPDX-License-Identifier: Apache-2.0
"""Choose draft depth from whole-batch work and measured ordinary decode."""

from collections import deque
from statistics import median

# Four decisions in a row with MTP below this fraction of ordinary decode
# (expected tokens per ms) park at once instead of after 16 losing ones.
_CLEAR_LOSS = 0.9


class BatchPolicy:
    """A policy belongs to one UID cohort, never to a model or single row.

    Costs include the complete batch call. Acceptance is pooled by conditional
    prefix reach, so rows with different accepted lengths contribute once each.
    Ordinary decode supplies its own baseline; depth-zero MTP is not a proxy.
    """

    def __init__(self, uids, max_depth, *, fixed=False, acceptance_by_depth=False):
        self.uids = tuple(uids)
        self.max_depth = max(1, max_depth)
        self.cur = self.max_depth
        # A fixed policy drafts max_depth every cycle: no ordinary-decode
        # calibration, no depth search and no parking. A fixed MTP depth and
        # a block drafter that verifies its whole block use it.
        self.fixed = bool(fixed)
        self.standard = deque(maxlen=8)
        self.standard_warmup = 2
        self.costs = {}
        self.acceptance = [0.6] * self.max_depth
        # Noncausal block widths have different proposal distributions, so
        # each width keeps its own curve; unmeasured positions stay None and
        # score from the pooled curve, which trains at every drafted width
        # (width-only priors left unrun widths unscoreable and locked
        # cohorts onto shallow depths).
        self.acceptance_by_depth = (
            {d: [None] * d for d in range(1, self.max_depth + 1)}
            if acceptance_by_depth
            else {}
        )
        self.losing = 0
        self.clear_losing = 0
        self.cooldown = 128
        self.remaining = 0
        # Measured MTP decisions since activation or the last park.
        self.decisions = 0
        self.elapsed_ms = 0.0
        self.last_probe_ms = 0.0
        self.last_seen = {}
        self.finished_at = None
        self.last_mode = None
        self._timing_interrupted = False

    def interrupt_timing(self):
        self._timing_interrupted = True

    def cycle_time_ms(self, mode, started, finished):
        # Between calls, the scheduler handles responses while the previously
        # dispatched MLX work may still run. Include that interval exactly
        # once, rather than timing only the host's wait inside next().
        begin = self.finished_at if self.last_mode == mode else started
        self.finished_at = finished
        self.last_mode = mode
        if self._timing_interrupted:
            # Prefill may have consumed pending decode work as well as wall time.
            self._timing_interrupted = False
            return None
        return max(0.0, (finished - begin) * 1000)

    def needs_standard(self):
        if self.fixed:
            return False
        return self.standard_warmup > 0 or len(self.standard) < 3 or self.remaining > 0

    def observe_standard(self, milliseconds):
        if milliseconds is not None:
            if self.standard_warmup:
                self.standard_warmup -= 1
            else:
                self.standard.append(max(1e-6, milliseconds))
        if self.remaining:
            self.remaining -= 1
            if not self.remaining:
                self.costs.clear()
                self.last_seen.clear()
                self.acceptance = [0.6] * self.max_depth
                for d in self.acceptance_by_depth:
                    self.acceptance_by_depth[d] = [None] * d
                self.cur = self.max_depth
                self.losing = 0

    def score(self, depth):
        if depth == 0:
            return len(self.uids) / median(self.standard) if self.standard else 0.0
        costs = self.costs.get(depth)
        if not costs:
            return 0.0
        expected = reach = 1.0
        acceptance = self.acceptance_by_depth.get(depth, self.acceptance)
        for position in range(min(depth, len(acceptance))):
            probability = acceptance[position]
            if probability is None:
                # An unmeasured position reads the pooled estimate.
                probability = self.acceptance[position]
            reach *= probability
            expected += reach
        return len(self.uids) * expected / median(costs)

    def _observe_acceptance(self, acceptance, counts, depth):
        # Conditional-prefix reach EMA; an unmeasured position starts from
        # the pooled estimate at the same position.
        for position in range(depth):
            reached = sum(count >= position for count in counts)
            if not reached:
                break
            rate = sum(count > position for count in counts) / reached
            if acceptance[position] is None:
                acceptance[position] = self.acceptance[position]
            acceptance[position] += 0.08 * (rate - acceptance[position])

    def observe_mtp(self, depth, accepted, milliseconds, *, stable, draft_depths=None):
        samples = {depth: accepted}
        if self.acceptance_by_depth and draft_depths is not None:
            # A transition can verify only a prefix of the original block.
            samples = {}
            for drafted, count in zip(draft_depths, accepted):
                samples.setdefault(drafted or depth, []).append(count)
        # Width curves train only once the calibration walk is done: during
        # the walk each width gets 2-3 samples that would shadow the pooled
        # prefix with flat-prior noise. The pooled curve always trains, so
        # a width that has not run yet still scores from a measured prefix.
        walk = any(
            len(self.costs.get(d, ())) < 3 for d in range(1, self.max_depth + 1)
        )
        for drafted, counts in samples.items():
            acceptance = self.acceptance_by_depth.get(drafted)
            if acceptance is not None and not walk:
                self._observe_acceptance(acceptance, counts, depth)
        self._observe_acceptance(self.acceptance, accepted, depth)
        if milliseconds is None or self.fixed:
            return
        self.elapsed_ms += milliseconds
        # A depth transition combines the previous asynchronous head and a new
        # CPU dispatch shape. Wait for a complete cycle at the same depth.
        if not stable:
            return
        self.costs.setdefault(depth, deque(maxlen=8)).append(max(1e-6, milliseconds))
        self.last_seen[depth] = self.elapsed_ms
        pending = [
            d for d in range(self.max_depth, 0, -1) if len(self.costs.get(d, ())) < 3
        ]
        if pending:
            self.cur = pending[0]
            return
        self.decisions += 1
        best = max(range(1, self.max_depth + 1), key=self.score)
        if self.score(best) > self.score(self.cur) * 1.03:
            self.cur = best
        self.losing = self.losing + 1 if self.score(best) <= self.score(0) * 1.03 else 0
        # A clear loss needs no long streak: at 8 rows on Qwen3.8-Flash-Next
        # the best depth reaches about 0.86x ordinary decode (M5 Ultra).
        clear = self.score(best) < self.score(0) * _CLEAR_LOSS
        self.clear_losing = self.clear_losing + 1 if clear else 0
        # Refresh stale alternatives without probing on every decision.
        if self.elapsed_ms - self.last_probe_ms >= 1000:
            rival = min(self.last_seen, key=self.last_seen.get)
            stale = self.elapsed_ms - self.last_seen[rival]
            if rival != self.cur and (
                self.score(rival) * 1.15 >= self.score(best) or stale >= 5000
            ):
                self.cur = rival
                self.last_probe_ms = self.elapsed_ms

    def should_park(self):
        return not self.fixed and (self.losing >= 16 or self.clear_losing >= 4)

    def park(self):
        self.start_parked(self.cooldown)
        self.standard_warmup = 2
        self.losing = 0
        self.clear_losing = 0

    def start_parked(self, cooldown, next_cooldown=None):
        """Decode ordinarily for ``cooldown`` steps before measuring MTP again.

        ``next_cooldown`` is the cooldown of the next park (default: twice
        this one, up to 4096 steps).
        """
        self.remaining = cooldown
        self.cooldown = (
            min(4096, cooldown * 2) if next_cooldown is None else next_cooldown
        )
        self.decisions = 0

    def held_up(self):
        """Whether MTP ran long enough in this cohort without being parked."""
        return not self.fixed and self.decisions >= 32


class ParkMemory:
    """Parking verdicts of one model that outlive a cohort.

    Every join or finish starts a new cohort policy, which would measure
    ordinary decode and MTP again from scratch. Where MTP just lost at k rows,
    a new cohort of at least k rows starts parked (MTP does not get cheaper
    per row as rows are added) for what is left of that park. The park runs
    on one clock per model, counted in batched decode steps of any cohort,
    so cohorts that come and go neither restart nor extend it: once it
    expires, the next cohort measures MTP again. A cohort where MTP held up
    for 32 measured decisions clears the verdicts it contradicts (those at
    its row count or fewer).
    """

    def __init__(self):
        self.clock = 0
        # rows -> (clock value the park ends at, cooldown of the next park)
        self._verdicts = {}

    def tick(self):
        """Count one batched decode step (ordinary or MTP) of this model."""
        self.clock += 1

    def seed(self, policy):
        if policy.fixed:
            return
        rows = len(policy.uids)
        live = [
            verdict
            for k, verdict in self._verdicts.items()
            if k <= rows and verdict[0] > self.clock
        ]
        if live:
            policy.start_parked(
                max(until for until, _ in live) - self.clock,
                next_cooldown=max(cooldown for _, cooldown in live),
            )

    def parked(self, policy):
        self._verdicts[len(policy.uids)] = (
            self.clock + policy.remaining,
            policy.cooldown,
        )

    def retired(self, policy):
        if policy.held_up():
            rows = len(policy.uids)
            for k in [k for k in self._verdicts if k <= rows]:
                del self._verdicts[k]
