# GPU-CR Onboarding: Start Here

> **Audience.** A new maintainer with solid C++ but little Linux systems
> programming and near-zero CUDA/PyTorch experience. This guide gives you the map;
> [`01-background-primer.md`](01-background-primer.md) gives you the vocabulary;
> docs `02`–`06` are per-subsystem references that walk every file function by
> function (purpose, callers, why it matters).

## What GPU-CR is

GPU-CR checkpoints and restores GPU applications. Its differentiator over plain
`cuda-checkpoint` or CRIU is that a checkpointed app's **VRAM usage drops to
zero** while the app's GPU pointers stay valid — so another workload can use the
GPU, and the original app later resumes exactly where it paused. It does this by:

1. **Injecting** a shared library (`vGPU-NVIDIA.so` / `vGPU-AMD.so`) into the
   application via `LD_PRELOAD` — no application changes.
2. **Recording** every GPU memory allocation the app makes, by interposing the
   allocation functions.
3. On checkpoint: **copying** all recorded VRAM to a hugepage-backed host staging
   buffer, then **releasing the physical VRAM while keeping the virtual addresses
   reserved** (the CUDA VMM trick — primer §6), and delegating the small opaque
   "control state" to a vendor tool (`cuda-checkpoint` on NVIDIA, CRIU on AMD).
4. On restore: re-allocating physical VRAM, re-mapping it at the same addresses,
   copying the bytes back, and letting the vendor tool revive control state.

The project builds on the FAST'26 GCR paper (see README §VII) and originated from
that codebase.

## The three operation modes

| Mode | CLI | Stops the process? | Vendor tool used | Key code path |
|---|---|---|---|---|
| **Single-GPU whole-process** | `cr_client -c/-r -p PID` | Yes (briefly) | cuda-checkpoint / CRIU | `vGPU.cpp: ckpt()/restore_ptr_and_content()` |
| **Multi-GPU whole-process** (NVIDIA) | `multi_cr_client -i/-c/-r -p PID1,PID2,…` | Yes | cuda-checkpoint | above + IPC teardown/rebuild phases in `ipc_hooks.cpp` |
| **Selective (memory-address)** | `cr_client -c/-r -s ptr:size,…` | **No** | none | `vGPU.cpp: ckpt_selective()/restore_ptr_and_content_selective()` |

The selective mode (v0.2.0) exists for multi-tenant LoRA fine-tuning: park one
tenant's ~1 GB of adapter/optimizer tensors while the process keeps serving other
tenants. It is the mode with the most active production use and the most open
issues — read `docs/design-memory-address-checkpoint.md` and
`docs/future-improvements.md` before touching it.

## Architecture at a glance

```
 ┌──────────────────────────────┐        signals (kill)         ┌──────────────────────┐
 │  Target process (e.g. vLLM)  │ ◄──────────────────────────── │  Coordinator CLI     │
 │  ┌────────────────────────┐  │                               │  cr_client (1 GPU)   │
 │  │ vGPU-*.so (LD_PRELOAD) │  │        control channel        │  multi_cr_client (N) │
 │  │  · alloc hooks         │  │ ◄──── shared memory file ───► │  (run by a human or  │
 │  │  · signal handlers     │  │   (signal_controls: msg +     │   the k8s Snapshot   │
 │  │  · ckpt/restore logic  │  │    selective region list)     │   Agent)             │
 │  │  · ipc_hooks (multi)   │  │                               └──────────────────────┘
 │  │  · nccl_hooks          │  │
 │  └───────┬───────┬────────┘  │        UDS + SCM_RIGHTS
 │          │       └───────────┼──────── fd re-exchange ─────► (peer worker processes,
 │          ▼                   │          (multi-GPU)           one per GPU)
 │   GPU backend (GPU.h)        │
 │   NVIDIA nv.cpp / AMD amd.cpp│
 └────┬──────────────┬──────────┘
      │              │ exec
      ▼              ▼
  GPU driver    cuda-checkpoint / criu        VRAM dump ──► hugetlbfs staging buffer
  (VMM APIs)    (opaque control state)                      (/mnt/huge-ckpt, SHM_SIZE)
```

Control flow for a checkpoint, end to end: the CLI writes a command word (and, for
selective ops, the region list) into the shared-memory control file, sends a signal
to the target PID, and polls for a `FINISH_MSG` ack. Inside the target, the signal
handler in `vGPU.cpp` reads the command, drives the vendor-neutral `GPU` interface
to dump/release memory, optionally execs the vendor tool, and writes the ack.

## Repository map and where each part is documented

```
src/                      the LD_PRELOAD library ("the preloader")
  vGPU.cpp                signal handlers, C/R orchestration, selective C/R ... 02
  common.h                signals, sizes, control structs .................... 02
  comm/                   shared-memory control channel ...................... 02
  backend/                hugepage staging-buffer backend .................... 02
  ipc_hooks.{h,cpp}       cuMem VMM/IPC/peer-access interception (multi-GPU) . 03
  ipc_fd_exchange.{h,cpp} UDS + SCM_RIGHTS fd exchange between workers ....... 03
  GPUs/                   vendor abstraction + NVIDIA/AMD implementations .... 04
  nccl_hooks.{h,cpp}      NCCL communicator tracking ......................... 04
coordinator/
  cr_client.cpp           single-GPU + selective CLI ......................... 05
  multi_cr_client.cpp     multi-GPU phased orchestrator ...................... 05
cuda-checkpoint/          vendored NVIDIA utility (binary + sample sources) .. 05
apps/vllm/                reference vLLM workloads/launchers ................. 05
adapters/nccl/            NCCL checkpoint plugin + upstream patches .......... 06
adapters/nvshmem/         NVSHMEM examples (no patch needed) ................. 06
CMakeLists.txt, Makefile, Dockerfile.build, cloudbuild-so.yaml,
.github/workflows/release.yml    build & release toolchain .................. 06
docs/                     design docs & this onboarding set (see below)
source/                   README images
```

## Suggested reading order

1. **`README.md`** — features, build, usage. Actually run the single-GPU flow if
   you have hardware; nothing teaches this system faster.
2. **[`01-background-primer.md`](01-background-primer.md)** — the OS/CUDA concepts
   (LD_PRELOAD, signals, shared memory, VMM, NCCL). Everything else assumes it.
3. **[`02-core-preload-library.md`](02-core-preload-library.md)** — `vGPU.cpp` is
   the heart: constructor, signal handlers, `ckpt()`/restore paths, staging.
4. **[`05-coordinator-clis-and-tools.md`](05-coordinator-clis-and-tools.md)** —
   the other side of the protocol; read together with 02.
5. **[`04-gpu-backends-and-nccl-hooks.md`](04-gpu-backends-and-nccl-hooks.md)** —
   how the vendor-neutral calls become CUDA/ROCm calls; cuda-checkpoint vs CRIU.
6. **[`03-ipc-interception.md`](03-ipc-interception.md)** — the multi-GPU IPC
   teardown/rebuild machinery (hardest part of the codebase; ~3 kLoC).
7. **[`06-adapters-and-build-system.md`](06-adapters-and-build-system.md)** — NCCL
   plugin + patches, NVSHMEM, and how releases are built.
8. The **design docs**, which double as the project's institutional memory:
   - `docs/design-memory-address-checkpoint.md` — selective C/R design (v0.2.0).
   - `docs/design-unrounded-dumps.md` — why dumps are no longer 2 MiB-rounded.
   - `docs/proposals/0001-verified-self-contained-selective-restore.md` — dump
     format v2 proposal, born from a real silent-corruption incident (2026-08-04).
   - `docs/future-improvements.md` — the de-facto maintainer backlog, with field
     evidence per item. **As incoming maintainer, treat this file as your issue
     tracker seed.**

## Dependencies you are responsible for

**Upstream (what GPU-CR depends on):**

| Dependency | Kind | Where it bites |
|---|---|---|
| CUDA driver + toolkit (12.x/13.x) | runtime | driver-API hooks in `ipc_hooks.cpp`; VMM behavior differs across driver versions; `cuda-checkpoint --action` needs recent drivers (r570/r580 feature sets — see `cuda-checkpoint/src/`) |
| `cuda-checkpoint` binary | vendored, `cuda-checkpoint/bin/` | invoked by `nv.cpp` for control state; keep in sync with driver capabilities |
| ROCm 6.x + custom-built CRIU with AMD plugin | external, not vendored | AMD path only works with a hand-built criu; `AMD_CKPT_DIR` env required |
| NCCL ≥ 2.28 source tree | build-time (adapter only) | `adapters/nccl/nccl_patches/` is a 3-file patch against upstream — every NCCL release can break it |
| NVSHMEM | build-time (examples only) | no patch; covered by cuMem hooks |
| hugetlbfs mount + reserved hugepages | operational | silent degradation if absent (future-improvements §5) |
| CMake ≥ 3.18, GCC, Linux | build | one vendor per build dir (`-DGPU_VENDOR=`) |

**Downstream (who depends on GPU-CR):**

- **The Kubernetes Snapshot Agent** (sibling project; `BACKEND_GPU_CR_MEMORY_ADDRESSES`
  backend) — the primary production consumer of the *selective* API. It shells out
  to `cr_client`, copies the staging buffer in/out of snapshot storage, and depends
  on: the CLI flags, the `shared_mem_fs` dump layout in the staging file, control
  file semantics, signal numbers (it patches them — future-improvements §6), and
  the `pid_map_<pid>` file. The 2026-08-04 corruption incident (proposal 0001) was
  a contract gap on exactly this seam — read Appendix A of the proposal.
- **Sibling projects in this workspace** (`../llm-d-rl-time-slicing`, `../open-rl`,
  the snapshot-agent docs in `..`) — the RL time-slicing demo stack that drives the
  agent, and thus GPU-CR, end to end.
- **vLLM users of whole-process C/R** — via `apps/vllm/` scripts and the README
  quick starts.
- Interface stability rule of thumb: `cr_client`/`multi_cr_client` flags, the
  signal numbers, the control-file struct (`signal_controls`), and the staging-file
  layout (`shared_mem_fs`) are all **de-facto public ABI**. Changing any of them
  requires coordinating a Snapshot Agent release.

## Build & smoke test in five commands

```bash
mkdir build && cd build
cmake -DGPU_VENDOR=NVIDIA ..            # or AMD; add -DSHM_SIZE_GB=40 for big dumps
make -j$(nproc)                         # → vGPU-NVIDIA.so, cr_client, multi_cr_client
LD_PRELOAD=$PWD/vGPU-NVIDIA.so python3 ../apps/vllm/serving_vllm_nvidia.py &
./cr_client -c -p $!                    # checkpoint; then ./cr_client -r -p $! to restore
```

See doc 06 for the full build matrix (NCCL/NVSHMEM adapters, cloud build, release
workflow) and README §V for the hugepage setup.

## The five invariants that keep this system correct

Internalize these before your first change; most historical bugs violated one:

1. **Virtual addresses are sacred.** Restore must land physical memory at the
   exact addresses the app already holds. Anything that perturbs address
   reservation across C/R breaks the app invisibly.
2. **The dump directory (`shared_mem_fs`) is trusted blindly by restore.** Today
   there is no versioning or checksum — restore will happily DMA stale or foreign
   bytes to the GPU (proposal 0001 fixes this; until it lands, ordering
   conventions like restore-before-checkpoint are load-bearing).
3. **Order matters in multi-GPU phases.** Teardown on *all* workers before any
   checkpoint; export before import on rebuild; init exactly once before anything.
   The phases exist because cross-process references must hit zero before the
   driver will release memory.
4. **The signal handler must always ack.** `cr_client` polls the control file for
   `FINISH_MSG` with no timeout; a handler that dies (today: `exit(-1)` paths)
   leaves the coordinator spinning forever (future-improvements §2).
5. **Hook coverage must be total.** Any allocation path the preloader misses
   (caching allocator internals, new CUDA APIs, cudaMallocAsync pools) is memory
   that silently won't be checkpointed. When CUDA/PyTorch/NCCL rev, re-audit the
   hook list first.

## Known sharp edges (as of 2026-08)

- Selective restore can transplant bytes across checkpoint groups when the shared
  staging buffer is reconstituted incompletely — root-caused, mitigated by
  ordering, proper fix specified in proposal 0001. Highest-priority open work.
- Interior/unknown pointers in selective requests: whole-allocation eviction,
  silent skips, and `exit(-1)` on handler failure (future-improvements §2).
- `CUDA_LAUNCH_BLOCKING=1` and `PYTORCH_NO_CUDA_MEMORY_CACHING=1` are currently
  required for the selective deployment and cost real performance
  (future-improvements §3/§7).
- Hugetlbfs silent-degradation and unpinned staging DMA (future-improvements §5).
- `SHM_SIZE` is compile-time and reserves ~27 GiB of hugepages per process
  (future-improvements §4).
- Git branch note: this clone's current branch is literally named `.invalid`;
  main development happens on `main` with feature branches (e.g.
  `unrounded-selective-dumps`, `memory-allocation` ↔ tag `v0.2.0`).
