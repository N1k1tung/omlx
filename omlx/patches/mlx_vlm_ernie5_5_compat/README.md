# ERNIE 5.5 Flash on the pinned mlx-vlm

The pinned mlx-vlm (`36494b4`, v0.8.1) ships the ERNIE 5.5 Flash Base model
and its ECHO drafter. This folder replaces them at load time with a copy that
also serves the Preview checkpoint, so oMLX needs no mlx-vlm fork.

## Source

`vendor/mlx_vlm/` holds three packages, byte for byte from
`arfung/mlx-vlm@5215b27e` (`ernie55-preview/8-decode-pipelining`):

- `models/ernie5_5_moe`
- `speculative/drafters/ernie5_5_dspark`
- `speculative/drafters/ernie5_5_mtp`

They start from mlx-vlm's own files (MIT, Copyright Prince Canuma and Neywa
Labs Inc.). The Preview loader and tokenizer are jonathan308's closed pull
request Neywa-Labs/mlx-vlm#2829.

## What `__init__.py` installs

That branch also edits shared mlx-vlm files. Here the same changes are
wrappers around the stock functions:

| mlx-vlm function | Wrapper |
|---|---|
| `fp8.transform_fp8_weights` | Repacks the Preview's MXFP4 experts and block FP8 projections |
| `utils.load_model_weights` | Evaluates the model one layer at a time |
| `utils.load_processor` | Loads the checkpoint's chat tokenizer, with the caller's `trust_remote_code` |
| `speculative.dflash._dflash_next_block_size` | Asks the drafter for its block size |
| `models.deepseek_v4.hyper_connection._hc_kernel` | Passes the collapse factors as fp32 |

`DRAFTER_KIND_BY_MODEL_TYPE` gains `ernie5_5_dspark`.

## Not carried over

Two edits of that branch serve mlx-vlm's own command line, which oMLX does not
use: the chat-template branch in `prompt_utils.get_chat_template` and the
quantization record in `utils.fetch_from_hub` for `mlx_vlm.convert`.
