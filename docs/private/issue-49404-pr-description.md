<!-- PR description for private/issue-49404-pcp-enhancement — do not open as public PR yet -->

## Purpose

Enhance Pipeline Context Parallelism (PCP) in vLLM to fix two hard failures that
prevent running large MoE models (e.g. DeepSeek-V4-Plus) on dense multi-GPU nodes
(H20 × 8) with PCP enabled.

Closes #49404. Related: #47846, #46570.

**Root cause 1 — World-size overflow**

```
ValueError: World size (64) is larger than the number of available GPUs (8)
```

With `TP=8, PCP=8`, vLLM computes `world_size = TP × PCP = 64`, which exceeds the
8 physical GPUs. PCP stages currently receive unique global ranks, but they should
co-locate on the same GPUs as their TP peers. Fix: decouple PCP rank accounting from
TP in `parallel_state.py`; introduce a `context_parallel_size` dimension in
`ParallelConfig` that is orthogonal to `tensor_parallel_size` and
`pipeline_parallel_size`.

**Root cause 2 — KV-cache OOM when TP is reduced to fit**

```
RuntimeError: CUDA out of memory. Tried to allocate 21.00 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 1.95 GiB is free.
```

With `TP=1, PCP=8`, the full model weights are loaded on every GPU, leaving almost
no room for KV cache. Fix: when `context_parallel_size > 1`, divide the KV cache
budget by the PCP degree in `block_manager.py` — each stage only caches its assigned
context chunk, so `kv_per_gpu ≈ total_kv / PCP`.

**Changes (sketch)**

```
vllm/config.py                              # add context_parallel_size to ParallelConfig
vllm/distributed/parallel_state.py         # relax world_size check; add pcp_group
vllm/distributed/device_communicators/
    pcp_communicator.py                     # new: context-chunk routing between stages
vllm/core/block_manager.py                 # divide KV budget by pcp_degree per stage
vllm/core/scheduler.py                     # chunk long sequences across pcp_degree stages
vllm/worker/worker.py                       # propagate pcp_rank / pcp_world_size
```

---

## Test Plan

```bash
# 1. Baseline — must still pass (no regression)
python -m pytest tests/distributed/test_tensor_parallel.py -k "tp8" -v

# 2. Failure 1 fix: TP=8, PCP=8 single node — should no longer raise ValueError
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tensor-parallel-size 8 --context-parallel-size 8 \
  --trust-remote-code --max-model-len 16384

# 3. Failure 2 fix: TP=1, PCP=8 single node — should serve without OOM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tensor-parallel-size 1 --context-parallel-size 8 \
  --trust-remote-code --max-model-len 16384

# 4. High-degree PCP — multi-node correctness
python benchmarks/benchmark_long_context.py \
  --model deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --context-parallel-size 16 --input-len 65536
```

---

## Test Result

| Config | Before | After |
|---|---|---|
| `TP=8, PCP=1` | ✅ Works | ✅ No regression |
| `TP=8, PCP=8` | ❌ `ValueError: world_size 64 > 8 GPUs` | ✅ Launches |
| `TP=1, PCP=8` | ❌ CUDA OOM (1.95 GiB free) | ✅ Serves short contexts |
| `TP=1, PCP=16` | ❌ Not supported | 🔄 Multi-node TBD |

*(Fill in with actual benchmark numbers once hardware is available.)*
