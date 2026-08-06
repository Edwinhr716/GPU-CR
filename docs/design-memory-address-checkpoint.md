# Design: Selective Memory-Address Checkpoint/Restore (GPU-CR v0.2.0)

Status: shipped in `v0.2.0` (tag `v0.2.0`, source-identical to branch `memory-allocation`).
Primary consumer: the Kubernetes Snapshot Agent's `BACKEND_GPU_CR_MEMORY_ADDRESSES` backend.

## 1. Problem statement

GPU-CR's original operation is whole-process checkpoint/restore: `cuda-checkpoint`
(NVIDIA) or CRIU (AMD) freezes the process and moves *all* of its GPU state. For the
workload that motivated this feature — multi-tenant LoRA fine-tuning, where one process
holds a large base model plus several tenants' small adapter + optimizer states — that
is the wrong granularity in three ways:

1. **Volume.** Evicting one tenant's ~0.1–1.5 GB of state should not move the ~8 GB base
   model with it.
2. **Availability.** Whole-process C/R stops the process. A trainer or vLLM sampler must
   keep serving other tenants while one tenant's state is parked.
3. **Memory pressure.** The point of eviction is to *free physical VRAM* for the active
   tenant. A copy alone doesn't help; the freed pages must actually return to the GPU,
   while the evicted tensors' pointers stay valid so the framework never notices.

The selective feature lets an external coordinator name exact memory regions
(`pointer:size` pairs) inside a running CUDA process, dump just the allocations
containing them to host memory, release their physical GPU backing, and later restore
the same virtual addresses — all without stopping the process or involving
`cuda-checkpoint`.

## 2. CLI additions

`cr_client` (in `coordinator/cr_client.cpp`) gained one flag; everything else is the
pre-existing surface:

```
cr_client [-i|-c|-r] -p <pid> [-m <criu_pid>] [-b] [-s ptr:size,...]
```

| Flag | New in this feature? | Meaning |
|---|---|---|
| `-s ptr:size,...` | **yes** | Comma-separated region list. Each entry is `pointer:size`; the pointer parses with `strtoull(…, 0)` so `0x…` hex or decimal both work. Size is bytes and must be non-zero. Max `MAX_SELECTIVE_REGIONS` = 4096 entries (`src/common.h:83`). |
| `-c` | no | Checkpoint. **With `-s`**: selective checkpoint (new code path, `cr_client.cpp:138`). Without `-s`: legacy whole-process dump via `cuda-checkpoint`/CRIU. |
| `-r` | no | Restore. **With `-s`**: selective restore (`cr_client.cpp:156`). Without: legacy path. |
| `-p` | no | Target PID. Must be the CUDA-context owner (see assumptions). |
| `-i`, `-m`, `-b` | no | Init handshake; CRIU pid override (AMD); buffer-only mode. Unrelated to selective mode. |

Exactly one of `-i|-c|-r` is required. Example, as issued by the snapshot agent:

```
cr_client -c -p 1234 -s 0x7f4a00000000:2490368,0x7f4a00400000:655360
cr_client -r -p 1234 -s 0x7f4a00000000:2490368,0x7f4a00400000:655360
```

**Caveat worth knowing:** on restore, the `-s` list is parsed and shipped to the target,
but the in-process handler (`restore_ptr_and_content_selective`, `src/vGPU.cpp:524`)
restores whatever the dump-file header describes — the region list is not consulted.
The dump is the source of truth; the restore-side `-s` argument is effectively
documentation.

## 3. Architecture

```
 cr_client (-c/-r -p PID -s …)                 target process (LD_PRELOAD=vGPU-NVIDIA.so)
 ─────────────────────────────                 ─────────────────────────────────────────
 parse regions ──► control file  ──signal──►   signal handler (CR_CKPT/CR_RESTORE)
   (selective_req struct)                        └─ ckpt_selective() / restore_…_selective()
 poll is_finished ◄───────────────────────────   └─ writes/reads dump file, flips finished
```

* **Interposition layer.** `vGPU-NVIDIA.so` is `LD_PRELOAD`ed into the workload. It
  replaces `cudaMalloc` with the CUDA VMM triple — `cuMemAddressReserve` +
  `cuMemCreate` + `cuMemMap` (`src/GPUs/NVIDIA/nv.cpp:354-464`) — which is the entire
  trick that makes selective release possible: virtual address and physical backing are
  decoupled, so physical pages can be unmapped and later re-created **at the same VA**.
  Every allocation is recorded in `allocated_memory` (ptr → *requested* size) and
  `global_handle_map`.
* **Control channel.** `ShareMemComm` maps a per-PID control file
  (`control-<pid>` under `/mnt/huge-ckpt`); `cr_client` writes the parsed
  `selective_cr_request` (fixed-size struct, `src/common.h:85-98`) plus a message word
  (`SELECTIVE_CKPT_MSG` / `SELECTIVE_RESTORE_MSG`), then sends `CR_CKPT_SIGNAL` /
  `CR_RESTORE_SIGNAL` and polls a `finished` flag at 1 ms.
* **Dump store.** Each CUDA process mmaps a `SHM_SIZE` (compile-time, default 25 GB,
  `-DSHM_SIZE_GB`) dump file at `/mnt/huge-ckpt/<id>` plus a `<id>-host` staging file
  and two 1 GB staging buffers (`src/backend/mmap_backend.cpp`). The code *assumes* the
  mount is hugetlbfs — it never passes `MAP_HUGETLB` — and logs
  "Hugepage shared memory mapped" regardless (see assumptions). An `EXPORT_FILE_PATH`
  env selects a plain-file backend (`ckpt-<id>.data`) instead.

## 4. Implementation details

### Checkpoint — `ckpt_selective()` (`src/vGPU.cpp:236`)

1. **Region → allocation resolution.** Each requested pointer is resolved to its
   *containing allocation* via `find_containing_allocation`; the resulting base
   pointers form a de-duplicated `std::set`. A pointer inside a tracked block selects
   the **whole block**; a pointer in no tracked block logs a warning and is skipped
   (`vGPU.cpp:280`). The requested *size* is not used for the copy — block granularity
   is the contract (interior pointers drag their whole allocation, see assumptions §6.5).
2. **Dump layout.** The dump file starts with a `shared_mem_fs` header — `file_num`,
   `current_offset`, and up to `MAX_FILE_NUM` = 4096 `(ptr, start_offset, size)`
   entries — followed by packed payload. In v0.2.0 each entry's size is
   `ROUND_UP_2MB(alloc_size)`.
3. **Copy pipeline.** Blocks stream GPU→host through two 1 GB pinned-ish staging
   buffers, double-buffered: `memcpyAsync` D2H into buffer *n* overlaps
   `memcpy_multi` (4-thread memcpy) of buffer *n−1* into the mmap'd dump file. A tail
   flush handles the final partial buffer.
4. **Physical release.** After the copy, every dumped block is released via
   `releasePhysicalMemory(ptr)` (`nv.cpp:170`): `cuMemUnmap` + `cuMemRelease` of the
   2MB-rounded block. The VA reservation and the `allocated_memory` entry survive, so
   `torch.Tensor.data_ptr()` values remain valid handles to now-unbacked memory.

### Restore — `restore_ptr_and_content_selective()` (`src/vGPU.cpp:524`)

1. **Remap.** For every dump-header entry: `remapPhysicalMemory(ptr, size)`
   (`nv.cpp:237`) re-runs `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` at the original
   VA (internally re-rounding size to 2 MB).
2. **Refill.** The same staging double-buffer pipeline runs in reverse
   (dump file → staging → `memcpyAsync` H2D). An assert enforces that file entries are
   contiguous in the dump in on-disk order.
3. The process continues without ever having observed the round-trip; subsequent
   kernels (including optimizer steps that *write* the restored tensors) run against
   the re-created physical pages. Write-after-restore was explicitly gate-tested.

### Failure behavior

Errors inside the signal handler (`exit(-1)` on stream/copy/space failures) terminate
the *workload*, and a workload that dies mid-operation leaves `cr_client` polling
forever — callers need their own timeout (the snapshot agent adds a 120 s
`GPU_CR_OP_TIMEOUT_SEC`). Unresolvable regions are skipped with a warning rather than
failing the operation.

## 5. What the consumer (snapshot agent) layers on top

Not part of GPU-CR, but part of the feature's real contract: the agent shells out to
`cr_client`, then copies the dump extents (`shared_mem_fs.current_offset`-limited) into
per-group snapshot storage (`snapshots/<group>/`), keyed by immutable group IDs
(tenant + version), and restores by copying a group's files back over the live dump
buffer before invoking `cr_client -r`. The dump file is a single shared scratch per
PID; groups exist only agent-side.

## 6. Assumptions and constraints

1. **`PYTORCH_NO_CUDA_MEMORY_CACHING=1` on the workload.** GPU-CR tracks whole
   `cudaMalloc` allocations. With PyTorch's caching allocator, one huge cudaMalloc pool
   contains many unrelated live tensors — snapshotting "one tensor" drags in its
   neighbors, and releasing its block would corrupt them. With caching off, one tensor
   ≈ one allocation and block granularity matches tensor granularity. This is a hard
   correctness requirement, at the cost of slower allocations and a 2MB-floor resident
   footprint per tensor.
2. **`CUDA_LAUNCH_BLOCKING=1`** is empirically required on hooked workloads (without it:
   illegal memory access / CUBLAS_ALLOC_FAILED). Eager mode only — no CUDA graphs, no
   `torch.compile` (the VMM hook and release/remap are invisible to captured graphs);
   vLLM consumers additionally need `enforce_eager` and
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the addresses and the signaled PID belong to
   the same process.
3. **Address stability is the caller's problem.** Snapshot only after the first
   `optim_step` (AdamW state materializes lazily); re-discover addresses after anything
   that reallocates (PEFT `delete_adapter`/`load_adapter`, `load_from_state`). GPU-CR
   validates nothing here — a stale address that happens to fall inside another live
   block will silently snapshot/release that block.
4. **2 MB granularity everywhere.** `ROUND_UP_2MB` (`src/common.h:22-23`) governs VMM
   allocation (a CUDA `cuMemCreate` floor), physical release, remap, and — in v0.2.0 —
   dump sizes. Consequences: resident footprint is inflated for small tensors, and
   dumps are amplified (measured 48× at opt-125m/rank-16, ~20× at 0.5B/rank-16, ~2.6×
   at 4B/rank-64). The dump-size half of this is removed by the unrounded-dumps change
   (see companion doc).
5. **Whole-allocation eviction.** An interior pointer selects its entire containing
   allocation. Safe under assumption 1; dangerous the moment multiple live tensors
   share an allocation. There is no base-pointer validation and no error for a
   non-base pointer.
6. **The dump directory must really be hugetlbfs** for the intended performance: the
   code mmaps whatever is mounted and succeeds silently on a plain directory (page
   cache, unpinned DMA — `cudaHostRegister` failure is tolerated with a "continuing
   without pinned memory" log, `nv.cpp:153`). hugetlbfs reservations are charged at
   mmap time: 25 GB + 2×1 GB ≈ 27 GiB of 2Mi hugepages **per CUDA process**, attached
   to the *file inode* — leaked dump files from dead processes keep the reservation.
7. **Signals.** Stock v0.2.0 uses `SIGUSR1`/`SIGUSR2`; Python workloads swallow these,
   so deployment builds patch to `SIGRTMAX-8`/`SIGRTMAX-7` at compile time (documented
   sed in the user guide). The control file is `0755` stock, `0777` in deployment
   builds so non-root workloads and the root agent can share it.
8. **Limits.** 4096 regions per request; 4096 dump-file entries; one selective
   operation at a time per process (`fs_mutex`); dump payload must fit `SHM_SIZE`.
   `pid_map_<pid>` (id↔pid mapping) is written via stdio and ends up empty on
   hugetlbfs — consumers fall back to `/proc/<pid>/maps`.
