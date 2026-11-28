"""How many tokens to draft each round, and when to draft by copying.

Verifying a block costs a fixed amount plus a few milliseconds per token, so a
proposal is worth verifying only while it is likely to be reached: every
proposal before it must have been accepted. ``DepthPolicy`` tracks how often
each depth is reached in the current request and drafts as deep as that pays.

``CopyIndex`` finds text the model is repeating from earlier in the request
(quoting a document, rewriting code). The continuation of the earlier
occurrence is then a better draft than the head's, and can fill the block.
"""

from typing import List, Optional, Sequence


class DepthPolicy:
    """Drafts while the chance of reaching a depth stays above ``threshold``.

    ``reach[i]`` estimates the probability that the first ``i`` proposals of a
    round are all accepted. A depth that was drafted updates from the outcome;
    a deeper one drifts slowly toward the value its neighbours imply, so the
    policy tries one token more when the current depth keeps being accepted
    in full, and retries a rejected depth only every several rounds.
    """

    # Reach probabilities assumed before a request has any history.
    PRIOR = (0.75, 0.5, 0.3, 0.2, 0.13, 0.09, 0.06)

    def __init__(
        self,
        max_proposals: int,
        threshold: float = 0.22,
        decay: float = 0.85,
        explore: float = 0.015,
        shallow: int = 3,
        shallow_threshold: float = 0.10,
    ):
        self.max_proposals = max(1, int(max_proposals))
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.explore = float(explore)
        # The first few proposals are cheap next to the round's fixed cost and
        # acceptance moves faster than the estimate, so they need less proof.
        self.shallow = int(shallow)
        self.shallow_threshold = float(shallow_threshold)
        prior = list(self.PRIOR)
        while len(prior) < self.max_proposals:
            prior.append(prior[-1] * 0.7)
        # Index i holds depth i; index 0 is the anchor, always reached.
        self.reach = [1.0] + prior[: self.max_proposals]

    def proposals(
        self, limit: Optional[int] = None, floor: Optional[float] = None
    ) -> int:
        """Proposals to draft. ``floor`` raises the reach a depth needs: when
        several requests share the verify forward its fixed cost is already
        spread, so an extra row has to earn more."""
        limit = self.max_proposals if limit is None else min(limit, self.max_proposals)
        count = 1
        for depth in range(2, limit + 1):
            needed = self.shallow_threshold if depth <= self.shallow else self.threshold
            if floor is not None:
                needed = max(needed, floor)
            if self.reach[depth] < needed:
                break
            count = depth
        return max(1, min(count, limit))

    def update(self, accepted: int, drafted: int) -> None:
        drafted = min(int(drafted), self.max_proposals)
        accepted = int(accepted)
        keep = self.decay
        for depth in range(1, drafted + 1):
            hit = 1.0 if accepted >= depth else 0.0
            self.reach[depth] = keep * self.reach[depth] + (1.0 - keep) * hit
        for depth in range(drafted + 1, self.max_proposals + 1):
            # Not drafted: assume it continues the trend of the depth before.
            previous = self.reach[depth - 1]
            before = self.reach[depth - 2] if depth >= 2 else 1.0
            step = previous / before if before > 1e-6 else 0.0
            implied = previous * min(step, 1.0)
            # A block accepted in full is direct evidence that one more token
            # would have been reached; anything else only earns a slow retry.
            full = accepted >= drafted and depth == drafted + 1
            rate = (1.0 - keep) if full else self.explore
            self.reach[depth] += rate * (implied - self.reach[depth])
        for depth in range(1, self.max_proposals + 1):
            # Reaching a depth implies reaching the one before it.
            self.reach[depth] = min(self.reach[depth], self.reach[depth - 1])


class CopyIndex:
    """Positions where the last ``order`` tokens occurred before."""

    def __init__(self, order: int = 6):
        self.order = int(order)
        self.tokens: List[int] = []
        self._last_end = {}
        # End position of the previous occurrence of the current suffix.
        self._match: Optional[int] = None

    def extend(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self.append(int(token))

    def append(self, token: int) -> None:
        self.tokens.append(int(token))
        end = len(self.tokens) - 1
        if end + 1 < self.order:
            self._match = None
            return
        key = tuple(self.tokens[end + 1 - self.order :])
        self._match = self._last_end.get(key)
        self._last_end[key] = end

    def continuation(self, count: int) -> List[int]:
        """Up to ``count`` tokens that followed the earlier occurrence."""
        if self._match is None or count <= 0:
            return []
        start = self._match + 1
        return self.tokens[start : start + count]
