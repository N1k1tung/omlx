"""CPU dispatch check: python tests/test_dflash_adaptive_cpu.py (no MLX import)."""

from __future__ import annotations

import ast
import logging
import random
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def _load(path, names, **namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("_ADAPTIVE_")
                for target in node.targets
            )
        )
    ]
    namespace.update(__name__=__name__, dataclass=dataclass, field=field, deque=deque)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_adaptive_predraft_overlaps_without_discarded_widths():
    # Execute the installed policy and actual drafter dispatch methods; replace
    # only tensor operations and the draft forward, keeping this check GPU-free.
    policy = _load(
        distribution("dflash-mlx").locate_file("dflash_mlx/engine/spec_epoch.py"),
        {"_AdaptiveBlockPolicy"},
    )["_AdaptiveBlockPolicy"]
    module = _load(
        Path(__file__).resolve().parents[1] / "omlx/speculative/dflash_drafter.py",
        {"_RowContext", "_concat_captured", "DFlashDrafter"},
        contextmanager=contextmanager,
        logger=logging.getLogger(__name__),
        mx=SimpleNamespace(
            concatenate=np.concatenate, int32=np.int32, async_eval=lambda *a: None
        ),
        set_verify_qmm_armed=lambda _: None,
        _q_arrays=lambda _: [],
    )
    drafter_class = module["DFlashDrafter"]
    rng = random.Random(0)
    modes = set()
    with patch.dict(
        "sys.modules",
        {"dflash_mlx.engine.spec_epoch": SimpleNamespace(_AdaptiveBlockPolicy=policy)},
    ):
        for pattern in ("full", "reject", "mixed"):
            drafter = drafter_class.__new__(drafter_class)
            drafter.block_size, drafter.verify_mode, drafter._rows = 8, "adaptive", {}
            drafter.model = SimpleNamespace(
                config=SimpleNamespace(draft_window_size=1024)
            )
            state = SimpleNamespace(uid=0)
            drafter._row(0).fed = 65_536
            assert drafter.block_limit(0) == 8
            forwards = []

            def forward(
                rows, *, block_size, forwards=forwards, drafter=drafter, **kwargs
            ):
                forwards.append(block_size)
                drafter._predraft = SimpleNamespace()
                return [(np.zeros((1, block_size - 1), dtype=np.uint32), [])]

            drafter._draft_batched = forward
            for _ in range(512):
                row_policy = drafter._row(0).policy
                modes.add(row_policy.mode)
                width = drafter.block_limit(0)
                before = len(forwards)
                queued = drafter.predraft(
                    None, state, [np.zeros((1, width, 1))], np.array([1]), np.array([7])
                )
                assert len(forwards) - before == int(queued)
                if pattern == "full":
                    assert queued  # Stable width 8 must retain fixed-mode overlap.
                    accepted = width - 1
                else:
                    accepted = 0 if pattern == "reject" else rng.randrange(width)
                drafter.record_cycle(
                    state.uid, width, accepted, rng.randrange(1, 4) * 1_000_000
                )
                if queued:
                    assert forwards[-1] == drafter.block_limit(0), (
                        "predraft dispatched a width discarded by this policy update"
                    )
    assert modes == {"full", "reduced", "probe"}


if __name__ == "__main__":
    test_adaptive_predraft_overlaps_without_discarded_widths()
    print(
        "Adaptive dispatch check passed: stable overlap, width transitions, 32k+ start."
    )
