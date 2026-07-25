# [Private] Feature: Enhance Pipeline Context Parallelism (PCP) in vLLM

**vLLM Issue:** [#49404](https://github.com/vllm-project/vllm/issues/49404)  
**Status:** Design / Investigation  
**Related:** [#47846](https://github.com/vllm-project/vllm/issues/47846), [#46570](https://github.com/vllm-project/vllm/issues/46570)  
**Branch:** `private/issue-49404-pcp-enhancement`  

---

## Problem Statement

PCP (Pipeline Context Parallelism) is now available via MRv2 (#47846, #46570), but the
current implementation has two hard failure modes when running large MoE models
(e.g. DeepSeek-V4-Plus) on dense multi-GPU nodes (H20-3E × 8):

### Failure 1 — TP + PCP world-size overflow

```
ValueError: World size (64) is larger than the number of available GPUs (8) in this node.
```

Reproducer: `--tensor-parallel-size 8 --pipeline-parallel-size 8`  
Root cause: the effective world size is computed as `TP × PCP = 8 × 8 = 64`, which
exceeds the 8 physical GPUs on the node. vLLM does not currently support
intra-node rank sharing across the TP and PCP dimensions.

### Failure 2 — KV-cache OOM when TP is reduced to fit

```
RuntimeError: CUDA out of memory. Tried to allocate 21.00 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 1.95 GiB is free.
```

Reproducer: `--tensor-parallel-size 1 --pipeline-parallel-size 8`  
Root cause: with TP=1 the model weights are fully replicated on each GPU; KV cache
has no room to grow, OOM follows.

---

## Requested Behaviour

The user wants two orthogonal modes to be supported simultaneously:

| Mode | Config | Semantics |
|------|--------|-----------|
| **Weight split + context split** | `TP=8, PCP=1` | Weights sharded across 8 GPUs; single context pipeline stage. Baseline working today. |
| **All-in-one group** | `TP=1, PCP=8` | Weights replicated, 8 pipeline stages each holding full model; context split across stages. Not working today due to OOM. |
| **High-degree PCP** | `TP=1, PCP=16/32/64` | Multi-node context parallelism; requires cross-node rank mapping. |

The fundamental ask is to decouple the world-size accounting of TP and PCP so that
`TP=N, PCP=M` is legal even when `N × M > local_gpu_count`.

---

## Root Cause Analysis

### 1. World-size check is additive, not aware of GPU sharing

`vllm/distributed/parallel_state.py` (and the underlying Megatron-style init) raises
when `world_size > torch.cuda.device_count()` on the current node. PCP stages are
assigned unique ranks, so `TP=8, PCP=8` tries to create 64 ranks on 8 GPUs.

Fix direction: introduce a **rank-reuse** or **virtual-rank** concept for pipeline
stages that co-locate on the same GPUs as their TP peers.

### 2. KV-cache allocation does not account for PCP degree

With `TP=1, PCP=8`, each GPU holds the full model weights (~130+ GiB on H20-139GiB),
leaving ~10 GiB for KV cache — enough for very short contexts but not production
workloads. The KV-cache allocator in `vllm/core/block_manager.py` does not reduce
the per-GPU reservation proportionally to `PCP degree`.

Fix direction: when PCP > 1 the KV cache should be split across pipeline stages
(each stage only caches its assigned context chunk), so `kv_per_gpu ≈ total_kv / PCP`.

### 3. No multi-node PCP dispatch

PCP degrees of 16/32/64 exceed any single node. The scheduler and the rank-init code
need to support cross-node virtual pipeline groups.

---

## Proposed Changes (sketch)

```
vllm/
  distributed/
    parallel_state.py        # relax world_size check; add pcp_group alongside tp_group
    device_communicators/
      pcp_communicator.py    # new: context-chunk routing between pipeline stages
  core/
    block_manager.py         # divide kv_cache budget by pcp_degree per stage
    scheduler.py             # chunk long sequences across pcp_degree stages
  worker/
    worker.py                # pass pcp_rank / pcp_world_size to each worker
  config.py                  # add pcp_size to ParallelConfig; validate constraints
```

### Config change

```python
# Current
ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=1)

# Proposed additions
ParallelConfig(
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    context_parallel_size=8,   # new — PCP degree; orthogonal to TP/PP
)
```

---

## Open Questions

1. Should PCP be a first-class dimension alongside TP and PP, or implemented as a
   virtual PP layer on top of MRv2?
2. How does chunked-prefill interact with PCP? Are they composable or mutually exclusive?
3. For `PCP=16/32/64`, what is the expected network topology (NVLink vs InfiniBand)?
4. What is the target max-model-len per PCP degree?

---

## Bisect / Test Plan

1. `TP=8, PCP=1` — baseline; must still pass.
2. `TP=1, PCP=8` on single H20 node — world-size fix must eliminate the ValueError;
   KV-cache fix must eliminate OOM for short contexts.
3. `TP=8, PCP=8` cross-node — cross-node rank init; measure context-throughput
   improvement vs `TP=8, PCP=1`.
4. `TP=1, PCP=32` multi-node — stress test; verify sequence chunking correctness
   with a long-context benchmark.

---

## References

- vLLM MRv2 design doc (internal)
- [#47846](https://github.com/vllm-project/vllm/issues/47846) — PCP support with MRv2
- [#46570](https://github.com/vllm-project/vllm/issues/46570) — Original PCP bug
- Megatron-LM context parallelism: [Megatron-LM #543](https://github.com/NVIDIA/Megatron-LM/pull/543)
