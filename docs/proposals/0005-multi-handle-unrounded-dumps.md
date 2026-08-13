# GEP-0005: Fix Unrounded Selective Dumps for Allocations Larger Than One VMM Handle

- **Status:** implemented (§3 proper fix; validated 2026-08-12, see
  Implementation History)
- **Authors:** @Edwinhr716 (with Claude)
- **Creation date:** 2026-08-06
- **Tracking:** rank-64 trainer crash 2026-08-06 (reproducer below); branch
  `unrounded-selective-dumps`
- **Related:** `docs/design-unrounded-dumps.md` (the feature this fixes),
  GEP-0003 (dump verification — composes)

## Summary

The unrounded-dumps change (`ckpt_selective` records and copies `alloc_size`
instead of `ROUND_UP_2MB(alloc_size)`) is validated only for allocations
that fit within a single 2MB VMM granule. The first workload whose selective
regions exceed 2MB — Qwen3-4B rank-64 LoRA, whose fp32 AdamW moments for
gate/up projections are 9728×64×4B = 2,490,368 bytes — crashes the workload
on the first region:

```
[vGPU-SELECTIVE-CKPT] Saving VMM block: base_ptr=0x75e000000 size=2490368 (aligned=2490368)
CUDA error at src/GPUs/NVIDIA/nv.cpp:133 - invalid argument
```

nv.cpp:133 is the `cudaMemcpyAsync` inside `nv::memcpyAsync`. The same
allocation copies successfully at its rounded size (4MB) under the rounded
build (validated end-to-end at 4B/rank-64), and unrounded copies of sub-2MB
regions are validated at scale (0.5B/rank-16: 1008 regions; 4B/rank-48:
1501 regions, max region 1,867,776B). The failure boundary is precisely
"unrounded length + allocation spanning >1 granule."

This proposal adds a **minimal reproducer**, an **interim two-line
mitigation** (copy at granule-aligned length, record unrounded length), and
the **proper fix** (granule-chunked copies), so unrounded dumps become safe
for arbitrarily large selective regions.

## Motivation

Unrounded dumps are the highest-leverage optimization in the selective-CR
stack: measured 22–58× reduction in bytes moved, which converted tenant
switching from losing to winning against every alternative at every
measured scale. Today that win silently excludes any workload with a single
tensor ≥2MB — which includes every LoRA rank ≥ 54 on Qwen3-4B-class MLP
widths (intermediate 9728: rank 54 × 9728 × 4B > 2MiB), i.e. exactly the
production-like configurations the optimization targets. The current
failure mode is the worst kind: the workload process dies inside a signal
handler mid-checkpoint (taking its tenant registry with it), and the
external agent sees only a timed-out operation.

### Goals

- Selective checkpoint/restore correctness for regions of any size under
  unrounded dumps.
- Keep the dump extent unrounded (the entire point of the feature).
- A CI-runnable reproducer pinned to the exact failure boundary.

### Non-Goals

- Root-causing the driver-level reason `cudaMemcpyAsync` rejects the
  unrounded length across a granule boundary in the hooked VMM context
  (tracked as an investigation item; both fixes below are correct
  regardless of the answer).
- Caching-allocator compatibility (KEP-0004) or dump verification (GEP-0003).

## Proposal

### 1. Reproducer (test-first)

A standalone gate configuration that fails today and gates the fix:
Qwen3-4B + rank-64 LoRA trainer swap (1512 regions, first >2MB region
2,490,368B), plus a unit-level CUDA reproducer: hooked `cudaMalloc` of
2,490,368B, `cudaMemcpyAsync` DtoH of the unrounded length — expected to
reproduce the `invalid argument` in isolation and to pin the boundary
(2MB+1 vs 2MB) exactly.

### 2. Interim mitigation (two lines, ships immediately)

In `ckpt_selective`, copy at the granule-rounded length while recording and
advancing the dump by the unrounded length:

```c
uint64_t size = alloc_size;                      /* recorded in dump: unrounded */
uint64_t copy_size = ROUND_UP_2MB(alloc_size);   /* device read length: rounded  */
```

The device mapping is always granule-rounded (`cuMemAddressReserve`/
`cuMemMap` of `aligned_size` at allocation time), so reading the padding is
always in-bounds — it is exactly what the rounded build has always done.
The staging pipeline copies `copy_size` from the device but writes only
`size` bytes into the dump at the recorded offset. Restore is symmetric
(HtoD of the unrounded length is not affected — restores of sub-2MB
lengths from granule-aligned bases are validated; the >2MB restore path
gets the same treatment if the reproducer shows it shares the boundary).

Cost: DtoH bandwidth for up to 2MB−1 of padding per multi-granule region —
bytes moved GPU→staging, but **not** written to the dump, so store size and
agent copy costs keep the full unrounded win. For the rank-64 case: 42
regions × ~1.7MB padding ≈ 71MB extra DtoH per swap (~2% of the rounded
build's traffic).

### 3. Proper fix: granule-chunked region copies

Replace the single `memcpyAsync` per staging-chunk with a walk that never
issues a copy crossing a granule boundary with an unaligned length:

- Split each region into `[base, next_granule)`, `[granule, granule+2MB)`,
  …, `[last_granule, base+alloc_size)` segments.
- Each segment is ≤2MB and either granule-aligned in base or in end —
  matching the copy shapes the validated configurations already exercise.
- Dump layout unchanged (segments concatenate to the unrounded extent).

This removes the padding traffic of §2 and, more importantly, makes the
copy shape independent of whatever driver constraint underlies the failure.
§2 ships first because it is two lines against a crash in production
configurations; §3 replaces it within the same branch once the reproducer
is in CI.

## Test Plan

- Unit reproducer (§1) red→green on both fixes.
- Gate B at 4B/rank-64 (the original crash): PASS with bitwise tensor
  verification and optim-step-after-restore.
- Regressions: 0.5B/rank-16 and 4B/rank-48 gates (validated configs) —
  timings within noise, dump extents unchanged.
- Dump-extent assertion: rank-64 group size ≈ Σ alloc_size (unrounded),
  not Σ rounded.

## Drawbacks

- §2 reads up to one granule of padding per multi-granule region — wasted
  DtoH bandwidth (bounded, measured ~2% for the motivating case) and a
  subtle divergence between "bytes copied" and "bytes dumped" that future
  readers must understand (comment mandatory).
- §3 multiplies `memcpyAsync` calls for large regions (a 2.4MB region
  becomes 2 copies; a 100MB region becomes ~50). Launch overhead is
  microseconds per call and the staging pipeline already chunks at 1GB, but
  per-region bookkeeping grows.
- Neither fix explains the underlying driver behavior; if the true cause is
  elsewhere (e.g., in the interception layer), these fixes mask it for the
  copy path while it may resurface in other unaligned driver interactions.
  The reproducer is the mitigation for this drawback: it pins current
  behavior and will surface regressions.

## Alternatives

1. **Stay rounded for multi-granule regions only** (round the *dump* for
   regions >2MB). Rejected: reintroduces store amplification exactly on the
   largest regions, which dominate group size (the rank-64 case would give
   back most of the unrounded win in the store).
2. **Cap supported rank/tensor size, document the limit.** Rejected: the
   limit excludes the production-like configurations the demo results are
   built on, and the failure mode (signal-handler crash) is unacceptable
   even as a documented limit — at minimum a pre-checkpoint size check
   refusing the operation cleanly would be required, which is more code
   than §2.
3. **Root-cause first, fix once.** Rejected as sequencing: production
   configurations crash today; §2 is two lines and testable immediately.

## Implementation History

- 2026-08-04 — unrounded dumps validated at 0.5B/rank-16 (22–58× reduction).
- 2026-08-06 — first >2MB selective region (4B/rank-64) crashes at
  nv.cpp:133; boundary confirmed by rank-48 (max region 1.87MB) passing
  clean on the identical build; campaign pivoted to rank-48 as workaround.
- 2026-08-06 — this proposal.
- 2026-08-12 — §3 (granule-chunked copies) implemented: `granule_clamp()`
  in `src/vGPU.cpp`, applied to both selective copy loops (DtoH checkpoint
  and HtoD restore); full-checkpoint paths untouched. Built as
  `gpucr-so:gep0005-chunked-v1`.
- 2026-08-12 — validation on the rebuilt `dra-testing` nodepool
  (`rl-hugepages-l4-b`, L4): Gate B at 4B/rank-64 PASS with the chunked
  build — 1502 regions (42×2,490,368B per tenant), dump extent 1.46875 GB
  = Σ alloc_size (unrounded win kept), 1512/1512 tensors bitwise,
  optim-step-after-restore OK, per-checkpoint transfer within noise of the
  unchunked build. Regression at 4B/rank-48 PASS (1501 regions). NOTE: the
  original crash no longer reproduces on the current node image — the
  incident-era `unrounded-v1` .so also passed rank-64 (1,278 unrounded
  >2MB copies, zero errors). The 2026-08-06 crash node and its driver
  stack are gone (nodepools recreated); the driver constraint appears
  environment-specific, which is precisely why §3 makes the copy shape
  independent of it. The unit reproducer (§1) therefore cannot currently
  pin the boundary red→green; it should be re-armed if the failure ever
  resurfaces on another driver.
