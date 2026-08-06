# Design: Unrounded Selective Dump Sizes

Status: implemented and validated 2026-08-04 on branch `unrounded-selective-dumps`
(commit `7a131b4`, one functional line on top of v0.2.0). Companion to
`design-memory-address-checkpoint.md`.

## 1. Problem statement

In v0.2.0 the selective checkpoint dumps each allocation at
`ROUND_UP_2MB(alloc_size)` (`src/vGPU.cpp:288`). Because the standard deployment runs
with the caching allocator disabled (one tensor per allocation) and LoRA-scale tensors
are mostly far smaller than 2 MB, nearly every dumped block is dominated by rounding
padding — bytes that were never handed to the application and carry no information.

Measured amplification of the v0.2.0 behavior:

| Workload | Real state | Dumped | Amplification |
|---|---|---|---|
| opt-125m / rank-16 sampler slot (144 regions) | ~6 MB | 288 MB | ~48× |
| Qwen2.5-0.5B / rank-16 tenant (1008 regions) | ~93 MB | ~2.0 GB | ~22× |
| Qwen3-4B / rank-64 tenant (1502 regions) | 1.47 GB | ~3.8 GB | ~2.6× |

Every byte of padding is paid for four times: GPU→host DMA, CPU memcpy into the dump
file, agent-side copy into snapshot storage, and the same path in reverse on restore.
At 0.5B scale this made the snapshot switch *lose* to a plain disk round-trip; it also
inflated agent snapshot storage (a memory-cgroup OOM source when snapshots live on
tmpfs) and dominated swap latency at every scale.

The fix: dump exactly `alloc_size` bytes — the size the application actually requested
— and let the padding exist only where it is unavoidable (physical VRAM, a CUDA VMM
floor).

## 2. CLI changes

**None.** The `cr_client` flags, the `ptr:size` region format, the control-file
struct, and the signal protocol are all unchanged. The only externally visible
differences:

* Dump files are smaller; `shared_mem_fs` header entries now carry unrounded sizes and
  unaligned `start_offset`s. Consumers that treat `current_offset` as an opaque byte
  count (the snapshot agent does) need no change.
* Log lines such as `Saving VMM block: … size=311296 (aligned=311296)` now show the
  unrounded value in both positions.
* Reported operation sizes (`tot_size`, and anything downstream like
  `storage_bytes`) now reflect payload bytes, which **no longer equal** device bytes
  freed (still block-granular). Metrics consumers should treat them as two different
  quantities.

## 3. Implementation details

The functional change is one line in `ckpt_selective()`:

```c
size_t alloc_size = it->second;          // original cudaMalloc request size
uint64_t size = alloc_size;              // was: ROUND_UP_2MB(alloc_size)
```

Everything else already tolerates unrounded sizes — this is why the change is safe,
verified path by path:

1. **The tracking map stores unrounded sizes already.** The `cudaMalloc` hook records
   `allocated_memory[ptr] = size` (the caller's request, `src/GPUs/NVIDIA/nv.cpp:458`)
   while allocating the rounded size physically. `alloc_size` at the checkpoint site is
   the true byte count; the rounding there was gratuitous.
2. **Physical release never reads dump metadata.** The release loop calls
   `releasePhysicalMemory(ptr)` with no size (`src/vGPU.cpp:376-383`); it looks the
   block up and rounds internally (`nv.cpp:170`). VRAM-freeing behavior is bit-for-bit
   identical before/after.
3. **Restore's remap re-rounds internally.** `restore_ptr_and_content_selective` passes
   the header size to `remapPhysicalMemory` (`vGPU.cpp:541`), which applies
   `ROUND_UP_2MB` before `cuMemCreate`/`cuMemMap` (`nv.cpp:237-248`).
   `ROUND_UP_2MB(alloc_size)` equals the original block size, so the recreated mapping
   is identical.
4. **The staging pipeline is alignment-agnostic.** Both copy loops chunk with
   `min(size, STAGING_BUF_SIZE - buf_offset)`; the checkpoint tail flush copies exactly
   `buf_offset` bytes and asserts `des_offset + buf_offset == fs->current_offset`
   (`vGPU.cpp:366-370`); the restore-side contiguity assert (`vGPU.cpp:584`) only
   requires entries packed in order, which offset accumulation preserves for any sizes.
5. **Padding bytes come back uninitialized — and that is fine.** The
   `[alloc_size, ROUND_UP_2MB(alloc_size))` tail of each re-created block previously
   round-tripped through the dump; now it is fresh uninitialized memory. Those bytes
   were never returned by `cudaMalloc`, so no correct program can read them.

Note the change is *not* sub-block dumping: the whole allocation is still saved. Under
a caching allocator (one big cudaMalloc pool holding many tensors), `alloc_size` is the
pool size and neighbors are still preserved — semantics identical to v0.2.0, minus only
the VMM rounding tail.

## 4. Measured results

Same gates, nodes, models and ranks as the v0.2.0 baselines; only `vGPU-NVIDIA.so`
changed:

| Metric | v0.2.0 (rounded) | Unrounded | Δ |
|---|---|---|---|
| 0.5B/rank-16 trainer: bytes per swap | ~2.0 GB | 0.093 GB | ~22× less |
| 0.5B swap_out / swap_in / switch | 1941 / 971 / 2865 ms | 693 / 195 / 411 ms | 2.8–7× |
| 4B/rank-64 trainer: bytes per swap | ~3.8 GB | 1.47 GB | 2.6× less |
| 4B swap_out / swap_in / switch p50 | 3612 / 1674 / ~5300 ms | 1961 / 852 / 2306 ms | ~2× |
| Sampler dump (opt-125m/rank-16) | 288 MB | 4.9 MB | 58× less |
| Sampler steady-state switch | 193 ms | 53 ms | 3.6× |
| VRAM freed per parked tenant | 2087 MB / 3823 MB | identical | — (by design) |

Correctness validation: trainer gate bitwise-identical restore (1008/1008 at 0.5B,
1512/1512 at 4B), optim_step-after-restore passes, sampler temperature-0 determinism
8/8, and a full 8-round × 2-tenant live RL run at 4B with zero failures and both
tenants converging.

## 5. Assumptions

1. **Everything the v0.2.0 feature assumes still applies** (no caching allocator,
   `CUDA_LAUNCH_BLOCKING=1`, eager mode, address-stability rules, hugetlbfs mount,
   4096-entry limits, signal remaps in deployment builds).
2. **Dump size ≠ VRAM freed.** Physical release and remap remain 2MB-block-granular
   (CUDA floor). Anyone using operation size as a proxy for reclaimed VRAM must switch
   to the device-bytes metric.
3. **Dump-header consumers must not assume 2MB-aligned sizes or offsets.** The known
   consumer (snapshot agent extent-limited copy) reads `current_offset` as a plain
   byte count and was verified; any new tooling that parses `shared_mem_fs` must do
   the same.
4. **Hugepage reservations are unchanged.** The dump file is still mmap'd at the full
   compile-time `SHM_SIZE` up front, so the ~27 GiB-per-process hugetlbfs reservation
   stands regardless of how small dumps become. Shrinking that is a separate change
   (`-DSHM_SIZE_GB`, see future-improvements doc).
5. **The remaining latency floor is per-block driver work** (~1500 `cuMemUnmap`/
   `cuMemCreate`/`cuMemMap`/`cuMemSetAccess` call groups per 4B tenant, ~70–110 ms per
   release pass plus remap). Only allocation packing can remove it; unrounded dumps do
   not.
