## Summary
Fix DSpark launch failure on SM120 for `deepseek-ai/DeepSeek-V4-Flash-DSpark` when speculative decoding is enabled with `num_speculative_tokens=7`.

Root cause:
- The sparse indexer forced decode flattening for all non-SM100 devices when `next_n > 2`.
- On SM120, DSpark decode must use the native sparse MLA decode path for small decode token counts (e.g. 7).
- Flattening on SM120 routed execution into an incompatible path and triggered startup failure.

## What changed
- Updated sparse indexer decode-path selection so SM120 is treated as native multi-token decode capable (same as SM100), avoiding flattening for DSpark `next_n > 2`.
- Added regression coverage for the platform capability decision helper to ensure SM100/SM120 return native support and others return non-native support.

## Files changed
- `vllm/v1/attention/backends/mla/indexer.py`
- `tests/v1/attention/test_indexer_dcp_localize.py`

## Validation
Executed:
- `python -m py_compile vllm/v1/attention/backends/mla/indexer.py tests/v1/attention/test_indexer_dcp_localize.py`

Attempted but unavailable in this environment:
- `pytest` (module not installed in the active virtual environment)

## Why this is not duplicate work
- The issue remains open and the failure mode here is specifically SM120 + DSpark speculative decode flattening behavior.
- This change directly targets the failing path reported in issue #49418 and adds focused regression coverage for the platform decision logic.

## AI assistance disclosure
This change was implemented with AI assistance and reviewed locally before commit.
