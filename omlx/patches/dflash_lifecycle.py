# SPDX-License-Identifier: Apache-2.0
"""Lifecycle wrap for dflash-mlx's class-level monkey patches.

dflash-mlx patches linear-attention / attention ``__call__`` at the class
level (``cls.__call__ = speculative_call`` etc.) inside its hook installer
functions, and uses class attributes like ``_dflash_speculative_call_installed``
as idempotency guards. Those patches persist for the lifetime of the
Python process — engine teardown does not undo them. Two engines sharing
a Python class then see crossed-over state: a later Native MTP load after
a DFlash session ends up with the dflash hook on ``linear_attn.__call__``
and the MTP draft cycle crashes with
``TypeError: speculative_call() got an unexpected keyword argument 'n_confirmed'``
(issue #1388).

This module wraps each dflash hook installer so oMLX can:
  - capture the pre-dflash ``__call__`` before dflash overwrites it
  - on ``restore_dflash_class_patches()`` (called from ``DFlashEngine.stop()``),
    revert each touched class to that captured state and clear dflash's
    idempotency flag so a subsequent DFlash load can re-arm cleanly

The wrap is idempotent and runs once per process — typically at the
beginning of ``DFlashEngine.start()`` just before ``load_target_bundle``.

The wrap also arms a batch-cache guard on every class dflash patches
(issue #2252): while a DFlash engine is loaded its hooks sit on Python
classes that other engines share (``mlx_lm.models.qwen3_next``'s
``Qwen3NextAttention`` is the very class qwen3_5 / qwen3_5_moe import as
``Attention``), so a concurrent BatchedEngine decode of an unrelated
model reaches dflash's ``attention_call``, whose ``int(cache.offset)``
crashes on the per-row ``mx.array`` offset of a ``BatchKVCache`` with
"[convert] Only length-1 arrays can be converted to Python scalars".
The guard routes any cache with an ``mx.array`` offset (batch caches
dflash never owns) to the pre-dflash ``__call__``.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


# cls -> {"call": pre_dflash_call, "flag": dflash_idempotency_attr_name}
_DFLASH_BACKUP: dict[type, dict[str, Any]] = {}


def _wrap_installer(mod: Any, fn_name: str, flag_name: str) -> bool:
    """Wrap ``mod.fn_name`` so each first-time class touch is recorded.

    ``fn_name`` is a dflash hook installer that takes a single
    ``linear_attn``-like argument and rewrites its class's ``__call__``.
    ``flag_name`` is the per-class idempotency attribute that dflash
    sets when its hook is installed — we use it to detect "already
    patched" so we don't double-record a backup.
    """
    if getattr(mod, "_omlx_wrapped_" + fn_name, False):
        return True

    original = getattr(mod, fn_name, None)
    if original is None:
        return False

    def wrapped(module_target: Any) -> Any:
        cls = type(module_target)
        current = cls.__dict__.get("__call__")
        if getattr(cls, flag_name, False) and getattr(
            current, "_omlx_mtp_call_marker", False
        ):
            # A Lightning MTP self-heal replaced dflash's hook while the
            # idempotency flag stayed set (issue #2972): dflash's installer
            # would skip re-installing and the engine would run without its
            # speculative hook. Clear the stale flag and re-snapshot the
            # current (MTP) __call__ as the base to restore/fall back to.
            with suppress(AttributeError):
                delattr(cls, flag_name)
            _DFLASH_BACKUP[cls] = {"call": current, "flag": flag_name}
        if not getattr(cls, flag_name, False):
            # First time dflash installs on this class — snapshot the
            # current __call__ so restore can put it back unchanged.
            _DFLASH_BACKUP.setdefault(cls, {"call": cls.__call__, "flag": flag_name})
        result = original(module_target)
        if getattr(cls, flag_name, False):
            _install_batch_cache_guard(cls)
        return result

    setattr(mod, fn_name, wrapped)
    setattr(mod, "_omlx_wrapped_" + fn_name, True)
    return True


def _install_batch_cache_guard(cls: type) -> None:
    """Wrap the dflash-installed ``__call__`` with a batch-cache bypass.

    dflash's hooks only ever run with its own single-sequence caches,
    whose ``offset`` is a plain int. Batch caches (``BatchKVCache`` /
    ``BatchRotatingKVCache``) carry a per-row ``mx.array`` offset, and
    reaching dflash's hook with one crashes in ``int(cache.offset)``
    (issue #2252). Any cache with an ``mx.array`` offset therefore goes
    to the snapshotted pre-dflash ``__call__`` instead.

    Idempotent per installed hook; restore drops the guard together with
    the hook because it rewrites ``cls.__call__`` from the backup.
    """
    dflash_call = cls.__call__
    if getattr(dflash_call, "_omlx_dflash_batch_guard", False):
        return
    info = _DFLASH_BACKUP.get(cls)
    if info is None:
        # No snapshot means the installer ran outside the wrap; nothing
        # safe to fall back to, so leave the hook untouched.
        return
    pre_dflash_call = info["call"]

    def guarded_call(
        self: Any, x: Any, mask: Any = None, cache: Any = None, **kwargs: Any
    ) -> Any:
        # Resolve the fallback base dynamically: a Lightning MTP load may
        # legitimately swap the non-dflash implementation underneath this
        # guard while a DFlash engine stays resident (issue #2972).
        live = _DFLASH_BACKUP.get(cls)
        base = live["call"] if live is not None else pre_dflash_call
        if kwargs.get("n_confirmed") or isinstance(
            getattr(cache, "offset", None), mx.array
        ):
            return base(self, x, mask=mask, cache=cache, **kwargs)
        return dflash_call(self, x, mask=mask, cache=cache, **kwargs)

    guarded_call._omlx_dflash_batch_guard = True  # type: ignore[attr-defined]
    cls.__call__ = guarded_call  # type: ignore[method-assign]


def get_dflash_guard_base(cls: type) -> Any | None:
    """Return the fallback base owned by an armed dflash guard."""
    current = cls.__dict__.get("__call__")
    if not getattr(current, "_omlx_dflash_batch_guard", False):
        return None
    info = _DFLASH_BACKUP.get(cls)
    if info is None:
        raise RuntimeError("dflash guard has no fallback base")
    return info["call"]


def set_dflash_guard_base(cls: type, new_call: Any) -> None:
    """Replace the fallback base under an armed dflash guard."""
    current = cls.__dict__.get("__call__")
    info = _DFLASH_BACKUP.get(cls)
    if not getattr(current, "_omlx_dflash_batch_guard", False) or info is None:
        raise RuntimeError("dflash guard state changed during MTP patching")
    info["call"] = new_call


def _install_cache_serializer() -> None:
    from dflash_mlx.cache import codecs
    from mlx_lm.models.cache import KVCache

    original = codecs.serialize_target_cache
    if getattr(original, "_omlx_valid_kv", False):
        return

    def serialize(target_cache, *, clone=True):
        fa, gdn = [], []
        for entry in target_cache:
            if isinstance(entry, KVCache):
                if entry.keys is None:
                    state = None
                else:
                    keys, values = entry.keys_and_values()
                    if clone:
                        keys, values = codecs._clone_array(keys), codecs._clone_array(
                            values
                        )
                    state = (keys, values, int(entry.offset))
                fa.append(state)
                gdn.append(None)
            else:
                layer_fa, layer_gdn = original([entry], clone=clone)
                fa.extend(layer_fa)
                gdn.extend(layer_gdn)
        return tuple(fa), tuple(gdn)

    serialize._omlx_valid_kv = True
    codecs.serialize_target_cache = serialize
    _install_glm_dfa_serializer(codecs, original)


def _install_glm_dfa_serializer(codecs: Any, original: Any) -> None:
    """Teach dflash's cache serializer GLM-5.3's composite DSA cache entries.

    dflash-mlx's snapshot codec serializes bare ``KVCache`` (FA layers) and
    its own ``RecurrentRollbackCache`` (GDN layers) only; a GLM-5.3 DSA layer
    holds ``CacheList(KVCache, PoolingCache)``, so the stock codec rejects the
    whole snapshot with a ``TypeError`` and the GLM adapter had to disable
    the prefix cache entirely.

    The wrap extends the codec in place: a composite entry serializes its KV
    sub-cache into the snapshot's FA slot (exact-length rows plus the offset,
    matching the bare-KVCache contract that ``hydrate_target_cache`` and the
    prefill-time offset validation already enforce) and the pooling sub-cache
    into the GDN slot (the full 5-tuple pool state plus its undo tail, the
    same logical state ``trim`` consumes). Restores go through
    :func:`omlx.patches.dflash_glm5.hydrate_glm_dsa_cache`, which rebuilds the
    composite from those slots.

    Wrapped once per process alongside the bare-KVCache serializer patch;
    the wrap stays active across engine reloads because snapshots for other
    (non-composite) backends never reach the added branch.
    """
    if getattr(codecs.serialize_target_cache, "_omlx_glm_dfa", False):
        return

    def serialize(target_cache, *, clone=True):
        from mlx_lm.models.cache import CacheList

        from .dflash_glm5 import _is_glm_dsa_cache

        fa, gdn = [], []
        for entry in target_cache:
            if not _is_glm_dsa_cache(entry):
                layer_fa, layer_gdn = original([entry], clone=clone)
                fa.extend(layer_fa)
                gdn.extend(layer_gdn)
                continue
            components = tuple(getattr(entry, "caches", ()) or ())
            kv_cache = components[0] if components else None
            pool_cache = components[1] if len(components) > 1 else None
            if kv_cache is None or not hasattr(kv_cache, "state"):
                raise TypeError(
                    "GLM DSA cache entry has no serializable KV component"
                )
            if hasattr(kv_cache, "keys_and_values"):
                keys, values = kv_cache.keys_and_values()
            else:
                keys, values = kv_cache.state[:2]
            if keys is None:
                fa.append(None)
            else:
                if clone:
                    keys = codecs._clone_array(keys)
                    values = codecs._clone_array(values)
                offset = int(getattr(kv_cache, "offset", 0) or 0)
                if int(keys.shape[2]) != offset or int(values.shape[2]) != offset:
                    raise ValueError(
                        "GLM DFA KV cache is not exact-length (buffered rows would "
                        "desync the snapshot's offset)"
                    )
                fa.append((keys, values, offset))
            if pool_cache is None or not hasattr(pool_cache, "state"):
                gdn.append(None)
                continue
            pool_state = tuple(pool_cache.state)
            undo = getattr(pool_cache, "_undo", None)
            if undo is not None:
                pool_state = pool_state + tuple(undo)
            gdn.append(
                tuple(codecs._clone_array(a) if clone else a for a in pool_state)
            )
        return tuple(fa), tuple(gdn)

    serialize._omlx_glm_dfa = True
    codecs.serialize_target_cache = serialize
    _install_glm_dfa_hydrator(codecs)


def _install_glm_dfa_hydrator(codecs: Any) -> None:
    """Teach dflash's ``hydrate_target_cache`` GLM's composite DSA cache entries.

    ``hydrate_target_cache`` rebuilds per-layer caches from a snapshot's FA and
    GDN slots using the template cache's types; it only knows bare KVCache,
    RotatingKVCache and dflash's own recurrent cache. The wrap routes GLM DSA
    composites (``CacheList(KVCache, PoolingCache)`` templates) through
    :func:`omlx.patches.dflash_glm5.hydrate_glm_dsa_cache`, which restores the
    exact-length KV rows and the pooling state (including its undo tail) that
    :func:`_install_glm_dfa_serializer` captured.
    """
    if getattr(codecs.hydrate_target_cache, "_omlx_glm_dfa", False):
        return

    original_hydrate = codecs.hydrate_target_cache

    def hydrate(snapshot: Any, template_cache: list[Any]) -> list[Any]:
        from mlx_lm.models.cache import CacheList

        from .dflash_glm5 import hydrate_glm_dsa_cache

        # Composite DSA templates reject inside dflash's stock hydrate before
        # the per-layer loop returns anything; dispatch them entirely here.
        rebuilt: list[Any] = []
        for index, tmpl in enumerate(template_cache):
            if isinstance(tmpl, CacheList):
                rebuilt.append(hydrate_glm_dsa_cache(snapshot, tmpl, index))
                continue
            rebuilt.append(None)
        if all(entry is not None for entry in rebuilt):
            return rebuilt
        hydrated = list(original_hydrate(snapshot, template_cache))
        for index, (entry, tmpl) in enumerate(zip(hydrated, template_cache)):
            if entry is None or isinstance(tmpl, CacheList):
                hydrated[index] = (
                    rebuilt[index]
                    if rebuilt[index] is not None
                    else hydrate_glm_dsa_cache(snapshot, tmpl, index)
                )
        return hydrated

    hydrate._omlx_glm_dfa = True
    codecs.hydrate_target_cache = hydrate


def install_dflash_lifecycle_wrap() -> bool:
    """Monkey-patch dflash's hook installers to record pre-dflash class state.

    Safe to call repeatedly — each installer is wrapped at most once.
    Returns True if at least one backend's installers were wrapped.
    """
    _install_cache_serializer()
    wrapped_any = False

    try:
        from dflash_mlx.engine import target_qwen_gdn as _qwen_gdn
    except ImportError:
        logger.debug("dflash_mlx.engine.target_qwen_gdn not importable")
    else:
        wrapped_any |= _wrap_installer(
            _qwen_gdn,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )
        # dflash 0.1.7 renamed the Qwen full-attention installer from
        # ``_install_split_full_attention_hook`` to ``_install_full_attention_gqa_hook``
        # (target_qwen_gdn). Without wrapping the new name the pre-dflash
        # ``Attention.__call__`` is never snapshotted, so a DFlash -> MTP
        # transition leaves dflash's hook on the class and the MTP draft
        # cycle crashes with "[convert] Only length-1 arrays ..." on the
        # per-row batched ``cache.offset`` (issue #1510).
        wrapped_any |= _wrap_installer(
            _qwen_gdn,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )

    try:
        from dflash_mlx.engine import target_gemma4 as _gemma4
    except ImportError:
        logger.debug("dflash_mlx.engine.target_gemma4 not importable")
    else:
        wrapped_any |= _wrap_installer(
            _gemma4,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )

    if wrapped_any:
        logger.debug("dflash lifecycle wrap installed")
    return wrapped_any


def restore_dflash_class_patches() -> None:
    """Revert every dflash-touched class to its pre-dflash ``__call__``.

    Also clears the dflash idempotency flag on each class so a later
    DFlash engine load can re-install its hook freshly. Empties the
    backup table.
    """
    restored = 0
    for cls, info in list(_DFLASH_BACKUP.items()):
        try:
            cls.__call__ = info["call"]
        except Exception as exc:
            logger.debug("restore failed for %s: %s", cls, exc)
            continue
        flag = info["flag"]
        if flag in cls.__dict__:
            with suppress(AttributeError):
                delattr(cls, flag)
        restored += 1

    _DFLASH_BACKUP.clear()
    # GLM-5.3's KDA adapter owns a separate class hook because dflash-mlx has
    # no upstream GLM module to wrap. Restore it at the same lifecycle
    # boundary so a later VLM/MTP GLM load never inherits it.
    try:
        from .dflash_glm5 import restore_glm5_dflash_class_patches

        restored += restore_glm5_dflash_class_patches()
    except Exception:
        logger.debug("GLM-5.3 dflash class restore skipped", exc_info=True)
    if restored:
        logger.info("dflash class patches restored on %d class(es)", restored)


def get_backup_classes() -> list[type]:
    """Return classes currently in the backup table — used by tests."""
    return list(_DFLASH_BACKUP.keys())
