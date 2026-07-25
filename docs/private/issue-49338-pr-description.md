# [Bugfix] Fix HummingLinearKernel passing wrong dtype for asymmetric zero-point quantization

## Summary

Fixes CI failure introduced by PR #46390: GSM8K score for
`nm-testing/Qwen3-4B-mixed-quant-RTN-wNaM` (humming, act-int8) degraded to
0.68 against a threshold of 0.85.

## Why This Is Not a Duplicate

Open PRs for issue #49338 were checked against the vllm-project/vllm repo at
time of authoring:

```bash
gh pr list --repo vllm-project/vllm --state open --search "49338 in:body"
gh pr list --repo vllm-project/vllm --state open --search "humming asymmetric zero point"
```

No open PR addressing this root cause was found. The upstream patch #46528
mentioned in the issue targets a different layer of the stack.

## Root Cause

`HummingLinearKernel.process_weights_after_loading`
(`vllm/model_executor/kernels/linear/mixed_precision/humming.py`) builds a
`quant_config` dict that is passed to the Humming library's
`BaseWeightSchema.from_config`. The `"dtype"` field in this dict controls how
Humming interprets the packed weight integers during dequantization.

Before this fix the field was always:

```python
"dtype": "int" + str(self.config.weight_type.size_bits)   # e.g. "int4"
```

This is correct for **symmetric biased** formats (e.g. `uint4b8` / GPTQ-int4)
where Humming's `"int4"` path handles the implicit offset internally.

For **asymmetric RTN** models (e.g. `nm-testing/Qwen3-4B-mixed-quant-RTN-wNaM`):

- The checkpoint's `weight_type` is `scalar_types.uint4` (unsigned, range 0–15,
  no implicit bias).
- Explicit per-group zero-point tensors are present (`zero_points=True`).
- Humming must interpret each packed 4-bit value as **unsigned** so the correct
  formula `(uint4_value - zero_point) * scale` is applied.
- Passing `"int4"` makes Humming treat the same bits as signed int4 (range
  −8 to 7), producing completely wrong dequantized values that zero-point cannot
  fix.

## Fix

One line changed in `humming.py`:

```python
# Before
"dtype": "int" + str(self.config.weight_type.size_bits),

# After
weight_dtype_prefix = "uint" if self.config.zero_points else "int"
"dtype": weight_dtype_prefix + str(self.config.weight_type.size_bits),
```

When `zero_points=True` the weights are always stored as unsigned integers
(`scalar_types.uint4`, `scalar_types.uint8`), so `"uint"` is semantically
correct.  When `zero_points=False` the existing `"int"` prefix is preserved
(symmetric GPTQ/compressed-tensors behaviour is unchanged).

## Files Changed

- `vllm/model_executor/kernels/linear/mixed_precision/humming.py` — dtype
  prefix fix (6 lines changed)

## Tests Run

*(Requires a machine with the Humming kernel installed and a humming-capable GPU.)*

```bash
# Install deps
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Run the failing eval (threshold 0.85)
python benchmarks/run_gsm8k.py \
  --model nm-testing/Qwen3-4B-mixed-quant-RTN-wNaM \
  --num-questions 1319 --num-fewshot 5 \
  --server-args "--enforce-eager --max-model-len 4096 --linear-backend humming"

# Run existing humming unit tests (no GPU required — import checks)
.venv/bin/python -m pytest tests/kernels/test_humming.py -v
```

Expected: GSM8K score ≥ 0.85 after fix (was 0.6831 before).

## Model Evaluation Results

Not runnable locally (no humming-capable GPU in this environment). The eval
config is `tests/evals/gsm8k/configs/humming/Qwen3-4B-mixed-quant-RTN-humming.yaml`.
The fix must be validated in the humming CI before merging.

## AI Assistance Statement

This fix was developed with AI assistance (Oz / Claude). The human submitter
has reviewed every changed line and understands the dequantization math.
