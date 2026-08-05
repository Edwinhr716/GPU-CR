# 05 — Coordinator CLIs, the cuda-checkpoint Tool, and the Example Apps

This chapter documents the *outside* of GPU-CR: the command-line programs that an
operator (or an automated agent) runs against a GPU process that already has the
GPU-CR interposition library (`vGPU-NVIDIA.so` / `vGPU-AMD.so`) preloaded into it.

There are three groups of artifacts:

1. **The coordinator CLIs** in `coordinator/`:
   - `cr_client` — drives checkpoint/restore of a **single** GPU process, including
     the *selective memory-address* mode (dump/restore only specific allocations).
   - `multi_cr_client` — orchestrates a **multi-GPU** job (e.g., a tensor-parallel
     vLLM server with one worker process per GPU) through a phased protocol.
2. **The vendored NVIDIA `cuda-checkpoint` repo** in `cuda-checkpoint/` — a prebuilt
   NVIDIA utility (plus NVIDIA's sample sources) that freezes/unfreezes the CUDA
   driver state of a process. GPU-CR shells out to this binary; it does not link
   against it.
3. **The example/benchmark apps** in `apps/vllm/` — scripts that launch vLLM
   inference servers in configurations that GPU-CR can checkpoint.

**Background: what is the "target process" here?** GPU-CR works by *interposition*:
the environment variable `LD_PRELOAD` tells the dynamic linker to load
`vGPU-NVIDIA.so` into a program before any other library, so that the library's
versions of functions like `cudaMalloc` shadow the real CUDA ones. The preloaded
library also installs signal handlers. The CLIs in this chapter never touch the GPU
themselves — they *ask the target process to do the work on itself* by sending it a
signal, and they pass parameters/results through a shared memory-mapped control file.

**Background: signals and `kill()`.** A signal is a tiny asynchronous notification
the OS can deliver to a process (e.g., `SIGUSR1`, "user-defined signal 1"). The
`kill(pid, sig)` system call — despite the name — just *sends* a signal to the
process with ID `pid`; it only terminates the process if the signal's default action
is termination and no handler is installed. A process can install a *signal handler*
(a function the kernel calls asynchronously, interrupting whatever the process was
doing) with `signal()`/`sigaction()`. GPU-CR's preloaded library installs handlers in
its constructor (`src/vGPU.cpp:1078`), so when a CLI calls `kill(pid, CR_CKPT_SIGNAL)`
the target process jumps into GPU-CR code. A signal carries **no payload** beyond its
number — which is exactly why GPU-CR pairs each signal with a message written to
shared memory (next paragraph).

**Background: shared memory.** Two processes normally cannot see each other's memory.
POSIX shared memory (classically created with `shm_open()` + `mmap()`) lets them map
the *same* physical pages so a write by one is instantly visible to the other. GPU-CR
uses the moral equivalent: both the CLI and the target `open()` the same file —
`/mnt/huge-ckpt/control-<pid>` (or `$EXPORT_FILE_PATH/control-<pid>`) — and `mmap()`
it with `MAP_SHARED` (`src/comm/share_mem.cpp:29`). `/mnt/huge-ckpt` is expected to
be a *hugetlbfs* mount (a RAM-backed filesystem using 2 MB "huge" pages), so this is
shared memory in all but API name. The mapped region is interpreted as
`struct signal_controls` (`src/common.h:95`): a 32-bit message word plus an inline
`selective_cr_request` for the selective mode.

## The handshake, in one paragraph

Every operation between a CLI and a worker follows the same order, and the order
matters:

1. CLI writes a request constant (e.g. `CKPT_MSG` = 11) into `control->signal`
   (`send_msg`).
2. CLI calls `kill(pid, <matching signal>)`.
3. The worker's signal handler wakes up, reads `control->signal` (`recv_msg`) to
   learn *which* operation was requested (several operations share one signal
   number), performs the work, and writes `FINISH_MSG` = **0** back into
   `control->signal`.
4. The CLI has been polling `is_finished()` (i.e. `control->signal == 0`) every
   1 ms; it unblocks and proceeds.

The message must be written *before* the signal is sent (otherwise the handler could
read a stale word), and completion is signaled by the same word returning to 0.
There are no timeouts anywhere in this loop — see "Gotchas".

## Sequence diagram: single-GPU checkpoint (`cr_client -c`)

```
 cr_client (CLI)                      target process (LD_PRELOAD=vGPU-NVIDIA.so)
 ---------------                      ------------------------------------------
   |  mmap /mnt/huge-ckpt/control-<pid>  (both sides share this page)
   |
   |  control->signal = CKPT_MSG (11)
   |------------------------------------->|   (visible via shared mapping)
   |  kill(pid, SIGUSR1)                  |
   |------------------------------------->|  cr_signal_handler(SIGUSR1)
   |                                      |    msg = recv_msg()      == CKPT_MSG
   |  poll: control->signal == 0 ?        |    syncAllKernels()
   |  (every 1 ms)                        |    ckpt(): copy ALL tracked GPU
   |                                      |      allocations -> /mnt/huge-ckpt/<id>
   |                                      |    disable P2P peer access
   |                                      |    control->signal = FINISH_MSG (0)
   |<-------------------------------------|
   |  fork/exec:                          |
   |  cuda-checkpoint --toggle --pid <pid>|  (CUDA driver: lock APIs, drain work,
   |------------------------------------->|   copy VRAM->host, release GPU)
   |  "Checkpointing done"                |  process still alive, GPU-free;
   v                                      v  now CRIU-able if desired
```

Restore (`cr_client -r`) runs the same handshake with `RESTORE_MSG`/`SIGUSR2`, then a
second `cuda-checkpoint --toggle` (see the ordering caveat in "Gotchas").

## Sequence diagram: multi-GPU phased flow (`multi_cr_client`)

```
 multi_cr_client                    worker 1..N (one process per GPU)
 ---------------                    --------------------------------
 CHECKPOINT (-c):
   | Phase 1: IPC Teardown
   |  for each w: send IPC_TEARDOWN_MSG; kill(w, SIGRTMAX-1)   --> all workers
   |  then wait for ALL to write FINISH_MSG                    <-- (unmap peers'
   |                                                                cuMem imports)
   | Phase 2: Data Dump
   |  for each w: send CKPT_MSG; kill(w, SIGUSR1)              --> all workers
   |  wait for ALL FINISH_MSG                                  <-- (GPU -> host dump)
   |
   | Phase 3: CUDA Control Freeze     (no signals; external tool)
   |  parallel fork/exec per worker:
   |    toggle mode:  cuda-checkpoint --toggle --pid <w>
   |    action mode:  --action lock   (all), then --action checkpoint (all)
   v

 RESTORE (-r):
   | Phase 1: CUDA Control Unfreeze   (external tool; must be FIRST)
   |    toggle mode:  cuda-checkpoint --toggle --pid <w>
   |    action mode:  --action restore (all), then --action unlock (all)
   |
   | Phase 2: Data Restore
   |  broadcast RESTORE_MSG + SIGUSR2 to all; wait all         <-- (host -> GPU)
   |
   | Phase 3a: IPC Re-export
   |  broadcast IPC_EXPORT_MSG + SIGRTMAX-2 to all; wait all   <-- (workers write
   |                                                                export tables
   |                                                                into their shm)
   | Phase 3 (exchange): CLI itself mmaps every worker's dump shm and
   |  cross-copies each worker's export table into every OTHER worker's
   |  "peer block" (no signals involved)
   |
   | Phase 3b: IPC Re-import
   |  broadcast IPC_IMPORT_MSG + SIGRTMAX-2 to all; wait all   <-- (workers re-import
   v                                                                peers' memory)
```

The critical rule in every broadcast phase: **signal ALL workers before waiting for
ANY of them** — collective operations block inside NCCL/IPC barriers until every rank
has entered.

---

# `coordinator/cr_client.cpp` — single-GPU CLI

Absolute path: `/Users/edwinhernandez/timeslice/GPU-CR/coordinator/cr_client.cpp`

This is the original, single-process driver. It is compiled twice by CMake — once
per vendor — and the `#ifdef __HIP_PLATFORM_AMD__` blocks select AMD (CRIU-based) or
NVIDIA (cuda-checkpoint-based) behavior at *compile* time.

**Background: CRIU.** CRIU ("Checkpoint/Restore In Userspace") is an external Linux
tool that can freeze an *entire process* — memory, threads, open files, sockets —
and serialize it to disk, then recreate it later. It works from the outside using
kernel facilities in the ptrace family (ptrace is the debugger interface that lets
one process inspect and control another). CRIU knows nothing about GPUs, which is
why the GPU state must be moved off the device first (by GPU-CR's dump plus
cuda-checkpoint on NVIDIA, or by AMD's CRIU plugin path here).

### `get_cuda_checkpoint_path()`

- **Purpose:** Locate the vendored `cuda-checkpoint` binary relative to the running
  executable. It reads `/proc/self/exe` — a symlink the kernel maintains for every
  process pointing at its own executable file (**Background:** `/proc` is a virtual
  filesystem where the kernel publishes live process information as fake files;
  `readlink()` resolves a symlink to the path it points at). From the executable's
  directory it builds `<dir>/../cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint`
  and checks it is executable with `access(..., X_OK)`.
- **Called by:** `main()` in this file only (checkpoint and restore branches).
  Verified by grep; `multi_cr_client.cpp` has its own, more elaborate copy.
- **Why it matters:** If the binary is missing it *falls back to the bare string*
  `"cuda-checkpoint"`, i.e. whatever is on `$PATH`, printing only a warning. A wrong
  or missing binary therefore does not stop the run — it fails later, quietly (see
  Gotchas: exit codes).

### `parse_selective_regions(const char* spec, selective_cr_request* req)`

- **Purpose:** Parse the `-s` argument, a comma-separated list of `pointer:size`
  pairs (e.g. `0x7f4a00000000:2490368,0x7f4a00400000:655360`), into the fixed-size
  `selective_cr_request` struct that is shipped to the worker through the control
  file. Pointers parse with `strtoull(..., 0)` so hex (`0x…`) and decimal both work;
  a zero size or a missing `:` is a hard error; at most `MAX_SELECTIVE_REGIONS`
  (4096, `src/common.h:83`) entries.
- **Called by:** `main()` in this file (both the selective-checkpoint and
  selective-restore branches). Verified by grep.
- **Why it matters:** This is the entire client-side surface of the selective
  memory-address feature described in
  `docs/design-memory-address-checkpoint.md`. Note that the struct is copied *by
  value* into the shared control page (`comm->control->selective_req = req;`)
  before the message word is set — so the worker sees a complete request when the
  signal arrives.

### `main(int argc, char* argv[])`

- **Purpose:** Parse flags, set up the shared-memory channel, and run exactly one
  operation.
- **Command-line flags** (getopt string `"icrdbp:m:s:"`):
  - `-i` — init: send `INIT_MSG` + `CR_INIT_SIGNAL` (SIGRTMAX). The worker runs
    `init_CR()`: allocates its CR ID, creates/maps its dump shm and staging buffers,
    and maps the control file from its side. Run this once after the model is
    loaded, *before* any checkpoint.
  - `-c` — checkpoint. Without `-s`: full dump (`CKPT_MSG` + `SIGUSR1`), wait for
    finish, then run `cuda-checkpoint --toggle --pid <pid>` (NVIDIA) or `criu dump`
    (AMD). With `-s`: selective dump only (`SELECTIVE_CKPT_MSG` + `SIGUSR1`); no
    external tool is involved and the process keeps running.
  - `-r` — restore, mirror of `-c` (with `-s`: `SELECTIVE_RESTORE_MSG` + `SIGUSR2`).
  - `-p <pid>` — target PID (mandatory; `assert(pid != 0)`).
  - `-m <criu_pid>` — AMD only: the PID handed to `criu dump -t`. CRIU dumps a
    *process tree* rooted at that PID, so when the Python process is a child of a
    launcher shell you point `-m` at the tree root while `-p` stays the CUDA-owning
    process that receives signals. Defaults to `pid`.
  - `-b` — "buffer only": do the GPU-CR data dump/restore but *skip* the external
    tool (cuda-checkpoint / CRIU). Useful for benchmarking the copy pipeline alone.
  - `-s ptr:size,...` — selective region list (see above).
  - Exactly one of `-i|-c|-r` is required. (A `dump` variable and a `d` in the
    getopt string exist, but there is no `case 'd':`, so `-d` falls into `default:`
    and prints usage — dead vestigial code.)
- **Why it matters:** This file *is* the single-GPU control plane, and it encodes
  the canonical ordering: data dump happens **before** the cuda-checkpoint freeze
  (the worker must still be able to issue CUDA copies while dumping). Also note the
  AMD restore ordering: `criu restore` recreates the process (same PID) first, and
  only then is `RESTORE_MSG`+`SIGUSR2` sent to refill GPU memory — you cannot signal
  a process that does not exist yet.

---

# `coordinator/multi_cr_client.cpp` — multi-GPU phased orchestrator

Absolute path: `/Users/edwinhernandez/timeslice/GPU-CR/coordinator/multi_cr_client.cpp`

Multi-GPU jobs (e.g. vLLM with tensor parallelism) run one worker process per GPU,
and the workers share GPU memory with each other through CUDA IPC. Checkpointing
them independently would deadlock or corrupt that shared state, so this tool runs a
*phased* protocol across all workers.

**Background: NCCL and CUDA IPC.** NCCL is NVIDIA's collective-communication library
(all-reduce, broadcast, …) used by every multi-GPU ML framework. To exchange data,
each worker *exports* some of its GPU allocations and the peers *import* (map) them
into their own address spaces — that is CUDA IPC (inter-process communication for
GPU memory). Those cross-process mappings are precisely what `cuda-checkpoint`
cannot handle ("does not support … IPC memory", per its README), so GPU-CR tears
them down before freezing and rebuilds them after restore. Comments mentioning
"NCCL Suspend/Resume" describe the *old* design; the current code uses generic IPC
teardown/rebuild and needs no NCCL source modifications.

### Checkpoint phases and their rationale (`do_checkpoint`)

1. **Phase 1 — IPC Teardown** (`IPC_TEARDOWN_MSG` + `SIGRTMAX-1`, skipped with
   `-n`): every worker unmaps the GPU memory it imported from peers. Must be first:
   as long as any worker still maps another worker's memory, neither the dump nor
   the cuda-checkpoint freeze sees a self-contained process.
2. **Phase 2 — Data Dump** (`CKPT_MSG` + `SIGUSR1`): each worker independently
   streams its tracked GPU allocations to its host-side dump file — same code path
   as single-GPU. Must precede the freeze because the dump itself issues CUDA
   copies, which are locked once the process is frozen.
3. **Phase 3 — CUDA Control Freeze**: run `cuda-checkpoint` for every worker
   (in parallel, one `fork()`+`execvp()` per worker). Last, because after this the
   workers can no longer touch the GPU.

### Restore phases (`do_restore`) — exact mirror image

1. **Phase 1 — CUDA Control Unfreeze**: `cuda-checkpoint` first, because until the
   CUDA driver state is restored every CUDA call in the workers blocks — the data
   restore in Phase 2 could never run. A failure here is treated as **fatal**
   (`exit(EXIT_FAILURE)`), with a hint to try the other cuda-checkpoint mode.
2. **Phase 2 — Data Restore** (`RESTORE_MSG` + `SIGUSR2`): each worker remaps
   physical GPU memory at the original virtual addresses and copies data back.
3. **Phase 3 — IPC Rebuild**, three sub-steps (skipped with `-n`):
   - **3a Re-export** (`IPC_EXPORT_MSG` + `SIGRTMAX-2`): each worker re-exports its
     shareable allocations and writes an export table (`IpcRebuildShmBlock`) into a
     reserved slot of its own dump shm.
   - **exchange** (CLI-side, `exchange_ipc_export_info()`, no signals): the CLI maps
     all N dump shms and cross-copies every worker's exports into every other
     worker's "peer block". The workers cannot do this themselves — each can only
     see its own shm.
   - **3b Re-import** (`IPC_IMPORT_MSG` + `SIGRTMAX-2`): each worker reads its peer
     block and re-imports the peers' memory (via `pidfd_getfd`, keyed by the
     `owner_pid` recorded in each export entry). Import must come after *all*
     exports exist, hence the strict 3a → exchange → 3b ordering.

### Function-by-function

#### `get_cuda_checkpoint_path()` (static)
- **Purpose:** Like the `cr_client` version, but hardened: resolves the project root
  with `realpath()`, tries two candidate locations, `chmod`s the binary to 0755 if
  it exists but is not executable, and returns `""` (instead of a PATH fallback) on
  failure.
- **Called by:** `init_cuda_checkpoint_path()` only (grep-verified).
- **Why it matters:** Returning `""` lets callers fail loudly per-phase instead of
  silently depending on `$PATH`.

#### `parse_pids(const char* arg)`
- **Purpose:** Split `-p pid1,pid2,...` into `std::vector<int>`, dropping
  non-positive entries.
- **Called by:** `main()`.
- **Why it matters:** Order is explicitly *not* significant — the CR-ID-to-PID
  mapping is resolved by other means (see `exchange_ipc_export_info`).

#### `elapsed_sec(t0)`
- **Purpose:** Timing helper (seconds since `t0`).
- **Called by:** every phase function; purely for the printed timings.
- **Why it matters:** The printed per-phase timings are the primary performance
  instrumentation of this tool.

#### `setup_workers(const std::vector<int>& pids)`
- **Purpose:** For each PID, construct a `ShareMemComm` (mapping
  `control-<pid>`) and record a `WorkerCtx { pid, cr_id = index, comm, shm_ptr }`.
- **Called by:** `main()`, once, before any action.
- **Why it matters:** Note that `cr_id` here is just the index in the PID list —
  which may **not** match the worker's real CR ID (assigned by an atomic counter in
  the workers). The IPC exchange code deliberately does not rely on this field.

#### `open_worker_shm(int cr_id)`
- **Purpose:** `open()` + `mmap()` a worker's dump shm by CR ID:
  `$EXPORT_FILE_PATH/ckpt-<id>.data` if `EXPORT_FILE_PATH` is set, else
  `/mnt/huge-ckpt/<id>`; maps `SHM_SIZE` bytes (default 25 GB — cheap because it is
  a mapping, not a copy).
- **Called by:** `exchange_ipc_export_info()`.
- **Why it matters:** This is the only place where the *CLI itself* opens the
  workers' big dump buffers; everywhere else it only touches the tiny control files.

#### `get_my_block(void*)` / `get_peer_block(void*)`
- **Purpose:** Compute the addresses of the two `IpcRebuildShmBlock` structures that
  live at the end of the 2 MB-rounded `shared_mem_fs` header region of each dump
  shm: `my_block` (last slot) holds the worker's own exports; `peer_block`
  (second-to-last) is where the CLI deposits everyone else's exports.
- **Called by:** `exchange_ipc_export_info()`.
- **Why it matters:** The layout arithmetic
  (`ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock)*{1,2}`) must
  stay byte-identical to the worker-side layout in `vGPU.cpp`; if either side
  changes the struct or the rounding, the exchange silently reads garbage.

#### `exchange_ipc_export_info()`
- **Purpose:** The CLI-side "switchboard" between restore phases 3a and 3b: open shm
  files `0..N-1`, snapshot each `my_block`, then write into each shm's `peer_block`
  the concatenation of all *other* shms' export entries (capped at
  `IPC_MAX_EXPORTS_PER_PROC` with a warning).
- **Called by:** `do_restore()` only.
- **Why it matters:** It indexes shm files by 0..N-1 on the assumption that N
  workers were assigned CR IDs 0..N-1 by the shared atomic counter — which is why
  `launch_multi_gpu.sh` zeroes the stale `/mnt/huge-ckpt/control` counter file at
  launch. Correct cross-process attribution relies on the `owner_pid` stored inside
  each export entry, not on file order. Any failure to open a shm is fatal.

#### `broadcast_and_wait(uint32_t msg, int sig, const char* phase_name)`
- **Purpose:** The core phase primitive: write `msg` + `kill(pid, sig)` to **all**
  workers first, then poll all workers' `is_finished()` (1 ms sleep).
- **Called by:** `do_init()`, `do_checkpoint()` (phases 1–2), `do_restore()`
  (phases 2, 3a, 3b).
- **Why it matters:** The signal-all-before-wait-any order is *the* correctness rule
  for collective phases — worker A's handler may block on an internal barrier until
  worker B enters the same phase. A failed `kill()` (worker already dead) aborts the
  whole run; a worker that dies *after* being signaled hangs the CLI forever.

#### `init_cuda_checkpoint_path()` / `detect_action_mode_support()`
- **Purpose:** Lazily cache the binary path; detect r580 "action mode" by running
  `cuda-checkpoint --help` through `popen()` and grepping the output for
  `--action`.
- **Called by:** `detect_action_mode_support()` and the freeze/unfreeze functions
  call `init_cuda_checkpoint_path()`; `main()` calls
  `detect_action_mode_support()` to auto-enable action mode.
- **Why it matters:** On 580+ drivers, `--toggle` restore is known to fail with
  "OS call failed or operation not supported on this OS", so auto-detection
  defaults you onto the working path.

#### `cuda_ckpt_run(int pid, const char* args, const char* label)`
- **Purpose:** Run one `cuda-checkpoint <args> --pid <pid>` invocation via
  `fork()` + `execvp()` + `waitpid()` and return the child's exit code.
- **Called by:** `cuda_ckpt_all_parallel()` (from worker threads) and
  `cuda_ckpt_all()`.
- **Why it matters:** Uses fork/exec instead of `system()` deliberately —
  `system()` is not thread-safe and the parallel path runs one thread per worker.
  Unlike `cr_client`, the real exit code is captured and propagated.

#### `cuda_ckpt_all_parallel(args, phase)` / `cuda_ckpt_all(args, phase)`
- **Purpose:** Run the same cuda-checkpoint action for every worker — in parallel
  with `std::thread` (the default) or sequentially (legacy helper). Both return the
  number of failures.
- **Called by:** `cuda_checkpoint_toggle_all`, `cuda_checkpoint_freeze_action`,
  `cuda_checkpoint_restore_action` (the sequential variant is currently only a
  fallback and is not referenced by the phase functions — grep shows
  `cuda_ckpt_all(` called nowhere else).
- **Why it matters:** Freezing N workers serially multiplies pause time by N;
  parallelism keeps the checkpoint window short.

#### `cuda_checkpoint_toggle_all(phase, buffer_only)`
- **Purpose:** Legacy mode: one `--toggle` per worker (parallel). No-op when
  `buffer_only` (`-b`) or on AMD builds.
- **Called by:** `do_checkpoint()` and `do_restore()` when action mode is off.
- **Why it matters:** `--toggle` is a single opaque flip; if it fails midway there
  is nothing to roll back — contrast with action mode below.

#### `cuda_checkpoint_freeze_action(phase, buffer_only)` / `cuda_checkpoint_restore_action(...)`
- **Purpose:** The r580 four-verb protocol, split across checkpoint and restore:
  freeze = `--action lock` on all workers, then `--action checkpoint` on all;
  restore = `--action restore` on all, then `--action unlock` on all. On lock or
  checkpoint failure, it attempts `--action unlock` everywhere to leave workers
  runnable.
- **Called by:** `do_checkpoint()` / `do_restore()` when `g_use_action_mode` is set
  (via `-a` or auto-detection).
- **Why it matters:** The two-step split *across all workers* (lock everyone before
  checkpointing anyone) is the multi-GPU analogue of a distributed barrier: no
  worker starts draining while another can still submit collective work against it.

#### `do_checkpoint` / `do_restore` / `do_init`
- **Purpose:** The phase sequences described above; `do_init` is a single broadcast
  of `INIT_MSG` + `CR_INIT_SIGNAL`.
- **Called by:** `main()`.
- **Why it matters:** These functions are the authoritative statement of phase
  ordering. If single-GPU `cr_client` and this file ever disagree (they do — see
  Gotchas), trust this file's ordering.

#### `usage` / `main`
- **Purpose:** Flag parsing and dispatch.
- **Command-line flags** (getopt string `"icrnbap:h"`):
  - `-i` / `-c` / `-r` — exactly one required (init / checkpoint / restore).
  - `-p pid1,pid2,...` — comma-separated worker PIDs (required; at most
    `MAX_MULTI_GPU_PROCS` = 32).
  - `-a` — force r580 `--action` mode. If omitted, action mode is still
    auto-enabled when `--help` output advertises it.
  - `-b` — buffer only: skip all cuda-checkpoint phases.
  - `-n` — "no NCCL": skip IPC teardown/rebuild phases (single-GPU-like workers
    with no cross-process GPU state).
  - `-h` — usage.
- **Why it matters:** This is the entry point automation should call for any
  tensor-parallel workload.

---

# `cuda-checkpoint/` — vendored NVIDIA utility

Absolute path: `/Users/edwinhernandez/timeslice/GPU-CR/cuda-checkpoint/`

This directory is a vendored copy of NVIDIA's public `cuda-checkpoint` repository.
The only artifact GPU-CR actually *uses* is the prebuilt binary at
`cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint`; the `src/` files are NVIDIA's
demo programs, kept for reference. The binary is a thin CLI over functionality that
lives *in the NVIDIA display driver* — driver updates add capability without a new
binary.

### What the binary does (`cuda-checkpoint/README.md`)

`cuda-checkpoint` suspends ("checkpoints") or resumes the **CUDA state** of one
process, identified by PID. On suspend the driver: (1) locks all CUDA APIs that
could affect GPU state, (2) drains already-submitted GPU work, (3) copies device
memory into driver-managed *host* allocations, and (4) releases every GPU resource.
The process itself keeps running on the CPU — threads that call CUDA simply block
until resume. Because a suspended process holds no GPU handles at the OS level, a
CPU-only tool like CRIU can then checkpoint it fully. Resume is the mirror image:
reacquire GPUs, copy memory back *to the original addresses*, rebuild streams and
contexts, unlock the APIs.

Operations:

- `--toggle --pid <pid>` — flip between running and checkpointed (the only verb on
  older drivers; a blind flip with no rollback).
- `--action lock | checkpoint | restore | unlock --pid <pid> [--timeout ms]` —
  r570/r580 split of the toggle into four explicit states.
  **Background — the four states:** `lock` freezes the *API surface* (no new CUDA
  work can be submitted; in-flight work drains; a timeout avoids deadlocks);
  `checkpoint` moves device state to host and releases the GPU (process must
  already be locked); `restore` re-creates GPU state from the host copy (process
  stays locked); `unlock` reopens the API surface and lets blocked CUDA calls
  proceed. So a full cycle is lock → checkpoint → … → restore → unlock, and
  multi_cr_client performs exactly that across all workers.
- `--get-state --pid <pid>` — print the process's current checkpoint state.
- `--get-restore-tid --pid <pid>` — print the CUDA restore thread ID.

Documented limitations that shape GPU-CR's design: x64 only; **no UVM or IPC
memory** (hence multi_cr_client's IPC teardown phase and vLLM's
`disable_custom_all_reduce`); waits for submitted work; no error recovery
mid-operation.

### How GPU-CR invokes the binary (grep-verified)

| Call site | How | Notes |
|---|---|---|
| `coordinator/cr_client.cpp` (ckpt & restore branches) | `system("<repo>/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint --toggle --pid <pid>")` | Path from `get_cuda_checkpoint_path()`; falls back to `$PATH`. Only `ret < 0` (spawn failure) is checked — a non-zero exit from the tool is ignored. |
| `coordinator/multi_cr_client.cpp` (`cuda_ckpt_run`) | `fork()`+`execvp()` per worker, parallel threads | Uses `--toggle` or `--action lock/checkpoint/restore/unlock`; real exit codes collected. |
| `src/GPUs/NVIDIA/nv.cpp:269` (`nv::externalCheckpoint` / `externalRestore`) | `system("cuda-checkpoint --toggle --pid <pid>")`, `$PATH` only | **Dead code today**: grep finds no caller of `externalCheckpoint`/`externalRestore` outside their own definitions — the external tool is invoked from the CLIs, not from inside the preloaded library (confirmed by the comment at `vGPU.cpp` `CKPT_MSG` branch). |

### The sample sources (NVIDIA demos, not built by GPU-CR)

- **`src/counter.cu`** — the README's example workload: a tiny CUDA program with a
  `__device__ int counter = 100` and an `increment` kernel, wrapped in a UDP echo
  loop on port 10000. Each received packet increments GPU memory and replies with
  the value — so after a checkpoint/restore cycle, a reply of "one more than last
  time" proves GPU state survived. GPU-CR's vLLM example apps copy this UDP-probe
  pattern.
- **`src/example.sh`** — the end-to-end demo as a script: start `counter`, probe it,
  `cuda-checkpoint --toggle` (GPU released, verified via `nvidia-smi`), `criu dump`
  (process killed, image written to `demo/`), `criu restore`, `--toggle` again,
  probe again.
- **`src/r570-features.c`** — demonstrates the r570 additions: a parent forks a
  child that initializes CUDA *and NVML* (NVIDIA's management library), then the
  parent uses the **CUDA driver API equivalents** of the CLI
  (`cuCheckpointProcessGetState`) plus CRIU 4.0's `cuda_plugin.so` (which drives the
  checkpoint automatically during `criu dump`) to checkpoint/restore the child over
  a live Unix socketpair (`--ext-unix-sk`/`--inherit-fd`).
- **`src/r580-migration-api.c`** — demonstrates r580 **GPU migration** via the
  driver API: a self-checkpointing process builds a `CUcheckpointGpuPair` table
  mapping every GPU's UUID to the next GPU's UUID, then loops
  `cuCheckpointProcessLock/Checkpoint/Restore/Unlock` so each restore lands the
  context on a different physical GPU. Requires persistence mode.
- **`src/r580-migration-cli.c`** — the same migration demo using the CLI instead:
  fork/execs `cuda-checkpoint --action restore --device-map "oldUuid=newUuid,..."`.
  Interesting for GPU-CR's future: migration between GPUs without touching the
  application.

---

# `apps/vllm/` — example / benchmark applications

**Background: vLLM.** vLLM is a popular open-source inference server for large
language models: you give it a model (a directory of weights or a HuggingFace ID)
and it serves text-generation requests, managing GPU memory (KV cache) efficiently.
For GPU-CR it is simply the flagship *workload*: a long-lived Python process with
many gigabytes of GPU state worth checkpointing.

**Background: tensor and pipeline parallelism.** A model too large for one GPU is
split across several. *Tensor parallelism* (TP) splits each layer's matrices across
N GPUs, which then cooperate on every single forward pass (heavy NCCL traffic —
one worker process per GPU). *Pipeline parallelism* (PP) instead assigns different
*layers* to different GPUs, passing activations along like an assembly line. Either
way you get N worker processes that all must be checkpointed coherently — the reason
`multi_cr_client` exists.

### `serving_vllm_nvidia.py`
- **What it runs:** A minimal single-GPU vLLM server: `LLM(model=...,
  enforce_eager=True)`, then a UDP loop on port 10000 — receive a prompt, run
  `llm.generate`, print timing, send the generated text back. (Same probe pattern
  as NVIDIA's `counter.cu`: one packet before checkpoint, one after restore.)
- **Env vars:** `VLLM_MODEL` — model path/ID (defaults to the placeholder
  `/path/to/your/model`). The process must additionally be *launched* with
  `LD_PRELOAD=<build>/vGPU-NVIDIA.so` and typically `GPU_VENDOR=NVIDIA` for GPU-CR
  to be present — this script does not set those itself.
- **Why it matters:** `enforce_eager=True` is load-bearing: it disables CUDA Graphs,
  which capture GPU work in a form that both cuda-checkpoint and GPU-CR's
  release/remap trick cannot cope with.

### `serving_vllm_amd.sh`
- **What it runs:** `exec python3 -m vllm.entrypoints.openai.api_server --model
  $VLLM_Model --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 2048` — a
  standard OpenAI-compatible HTTP vLLM server on one AMD GPU. The `exec` matters:
  the shell *replaces itself* with Python, so the shell's PID (`$$`, echoed at
  startup) **is** the Python PID you pass to `cr_client -p`, and CRIU sees a clean
  single-process tree.
- **Env vars set by the script:** `CUDA_VISIBLE_DEVICES=""` (hide NVIDIA devices),
  `HIP_VISIBLE_DEVICES=0` (use AMD GPU 0), `GPU_VENDOR=AMD` (selects the AMD code
  path in vGPU.so), `VLLM_Model` from `$VLLM_MODEL`. The AMD flow also requires
  `AMD_CKPT_DIR` for `cr_client` (where CRIU writes images) — set that in the
  environment of the *client*, not this script.
- **Why it matters:** It is the reference recipe for making the "which PID do I
  signal / which PID does CRIU dump" question trivial on AMD.

### `serving_vllm_multi_gpu.py`
- **What it runs:** The multi-GPU counterpart of `serving_vllm_nvidia.py`:
  `LLM(model, tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
  enforce_eager=True, disable_custom_all_reduce=True, gpu_memory_utilization=...)`,
  plus the same UDP probe loop.
- **Flags / env vars:** `--model` (default `$VLLM_MODEL`), `--tp` (tensor-parallel
  size, default 1), `--pp` (pipeline-parallel size — note the default is **2**, so
  running it bare uses PP=2), `--port` (default 10000), `--max-model-len`,
  `--gpu-util` (default `$VLLM_GPU_UTIL` or 0.5). It *reports* `NCCL_CUMEM_ENABLE`
  and `LD_PRELOAD` at startup but relies on the launcher to have set them.
- **Why it matters:** Its two non-default LLM kwargs encode GPU-CR requirements:
  `enforce_eager=True` (no CUDA Graphs, as above) and
  `disable_custom_all_reduce=True` — vLLM's custom all-reduce shares buffers via
  `cudaIpcGetMemHandle`, which does not work on GPU-CR's VMM-backed
  (`cuMemCreate`/`cuMemMap`) allocations, so it must fall back to NCCL. The header
  comment also shows the intended operator loop: `pgrep -f serving_vllm_multi_gpu`
  to collect worker PIDs, then `multi_cr_client -i/-c/-r -p <pids>`.

### `launch_multi_gpu.sh`
- **What it runs:** A generic launcher: sets up the environment, then `exec "$@"` —
  so `./launch_multi_gpu.sh python serving_vllm_multi_gpu.py --tp 4` (or `torchrun
  ...`) runs the given command with GPU-CR injected.
- **What it does, in order:**
  1. Exports `CUDA_HOME` (default `/usr/local/cuda`) and prepends its `bin`/`lib`
     dirs to `PATH`/`LD_LIBRARY_PATH`.
  2. Locates `build/vGPU-NVIDIA.so` (or `vGPU-AMD.so`) relative to itself; errors
     out if not built.
  3. Warns if `/mnt/huge-ckpt` is not mounted, and — important — **zeroes the first
     4 bytes of `/mnt/huge-ckpt/control`**, the cross-process atomic CR-ID counter,
     so the next N workers get IDs 0..N-1 (which `multi_cr_client`'s IPC exchange
     step assumes).
  4. Exports the NCCL/vLLM env (below), then builds `LD_PRELOAD` as
     `vGPU.so:<existing preload>:<local libnccl.so.2>` — vGPU.so **must be first**
     so its hooks win; a locally built NCCL from `nccl_install/lib` is appended if
     present (and recorded in `CR_NCCL_LIB`).
- **Env vars that matter:**
  - `NCCL_CUMEM_ENABLE=1` — makes NCCL allocate through the cuMem (VMM) API, the
    allocation style GPU-CR's hooks can track, tear down, and rebuild. Must be set
    *before* vLLM/NCCL initializes; this is the single most load-bearing line.
  - `NCCL_P2P_DISABLE=1` — no direct GPU-to-GPU peer access (P2P driver state is
    another thing cuda-checkpoint cannot restore).
  - `NCCL_CUMEM_HOST_ENABLE=1`, `NCCL_DEBUG` (default WARN).
  - `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` and `VLLM_DISABLED_KERNELS=custom_all_reduce`
    — belt-and-suspenders env versions of `disable_custom_all_reduce=True`.
  - `GPU_VENDOR=NVIDIA` — vendor selection for vGPU.so.
- **Why it matters:** Every multi-GPU checkpoint bug report should start with "was
  it launched through this script?" — a worker started without `NCCL_CUMEM_ENABLE=1`
  produces IPC state that the teardown phase cannot see.

---

# Protocol reference

Signal numbers assume Linux glibc defaults: `SIGUSR1`=10, `SIGUSR2`=12,
`SIGRTMAX`=64 ("real-time" signals are an extra numbered range, 34–64, that
applications may use freely; unlike classic signals they queue instead of
coalescing). Defined in `src/common.h:37-45`. **Deployment caveat:** stock builds
use `SIGUSR1`/`SIGUSR2`, but Python runtimes often have their own handlers for
those; deployment builds patch them to `SIGRTMAX-8`/`SIGRTMAX-7` at compile time
(see `docs/design-memory-address-checkpoint.md` §6.7).

### Signals

| Constant | Value | Sent by | Handled by (in target) | Expected behavior |
|---|---|---|---|---|
| `CR_INIT_SIGNAL` | `SIGRTMAX` (64) | `cr_client -i`, `multi_cr_client -i` | `cr_signal_handler` (`vGPU.cpp:728`) | Run `init_CR()` (allocate CR ID, map dump shm + staging buffers), then `FINISH_MSG` |
| `CR_CKPT_SIGNAL` | `SIGUSR1` (10) | `cr_client -c` (with or without `-s`), `multi_cr_client` Phase 2 | `cr_signal_handler` | Dispatch on message: `CKPT_MSG` → full dump + P2P disable; `SELECTIVE_CKPT_MSG` → selective dump + physical release; then `FINISH_MSG` |
| `CR_RESTORE_SIGNAL` | `SIGUSR2` (12) | `cr_client -r`, `multi_cr_client` restore Phase 2 | `cr_signal_handler` | `RESTORE_MSG` → full remap+refill + P2P re-enable; `SELECTIVE_RESTORE_MSG` → selective remap+refill; then `FINISH_MSG` |
| `CR_IPC_TEARDOWN_SIGNAL` | `SIGRTMAX-1` (63) | `multi_cr_client` checkpoint Phase 1 | `cr_ipc_signal_handler` (`vGPU.cpp:829`) | Unmap imported peer memory; `FINISH_MSG` |
| `CR_IPC_REBUILD_SIGNAL` | `SIGRTMAX-2` (62) | `multi_cr_client` restore Phases 3a & 3b | `cr_ipc_signal_handler` | Dispatch on message: `IPC_EXPORT_MSG` → re-export + write my_block; `IPC_IMPORT_MSG` → read peer_block + re-import; then `FINISH_MSG` |
| `CR_IPC_VALIDATE_SIGNAL` | `SIGRTMAX-3` (61) | nothing in-repo (manual `kill -61 <pid>` diagnostic) | lambda in `init()` (`vGPU.cpp:1093`) | Log validation of all IPC mappings; **no** shm handshake |

(`CR_NCCL_SUSPEND_SIGNAL`/`CR_NCCL_RESUME_SIGNAL` are aliases of the IPC
teardown/rebuild signals, kept for backward compatibility.)

### Shared-memory message constants (`src/comm/comm.h`)

| Constant | Value | Written by | Read by | Paired signal | Response |
|---|---|---|---|---|---|
| `FINISH_MSG` | 0 | worker (all handlers, on completion) | CLI `is_finished()` poll | — | terminates the CLI's wait loop |
| `INIT_MSG` | 10 | `cr_client -i`, `multi_cr_client -i` | `cr_signal_handler` | `CR_INIT_SIGNAL` | init_CR, `FINISH_MSG` |
| `CKPT_MSG` | 11 | `cr_client -c`, `multi_cr_client` Phase 2 | `cr_signal_handler` | `CR_CKPT_SIGNAL` | full dump, `FINISH_MSG` |
| `RESTORE_MSG` | 12 | `cr_client -r`, `multi_cr_client` restore Phase 2 | `cr_signal_handler` | `CR_RESTORE_SIGNAL` | full restore, `FINISH_MSG` |
| `NCCL_SUSPEND_MSG` | 13 | *(legacy, no current writer)* | — | — | — |
| `NCCL_RESUME_MSG` | 14 | *(legacy, no current writer)* | — | — | — |
| `IPC_TEARDOWN_MSG` | 15 | `multi_cr_client` checkpoint Phase 1 | `cr_ipc_signal_handler` | `CR_IPC_TEARDOWN_SIGNAL` | teardown, `FINISH_MSG` |
| `IPC_EXPORT_MSG` | 16 | `multi_cr_client` restore Phase 3a | `cr_ipc_signal_handler` | `CR_IPC_REBUILD_SIGNAL` | write export table, `FINISH_MSG` |
| `IPC_IMPORT_MSG` | 17 | `multi_cr_client` restore Phase 3b | `cr_ipc_signal_handler` | `CR_IPC_REBUILD_SIGNAL` | import peers, `FINISH_MSG` |
| `SELECTIVE_CKPT_MSG` | 20 | `cr_client -c -s ...` | `cr_signal_handler` | `CR_CKPT_SIGNAL` | `ckpt_selective(&control->selective_req)`, `FINISH_MSG` |
| `SELECTIVE_RESTORE_MSG` | 21 | `cr_client -r -s ...` | `cr_signal_handler` | `CR_RESTORE_SIGNAL` | `restore_ptr_and_content_selective()`, `FINISH_MSG` |

The channel itself: `struct signal_controls { uint32_t signal;
selective_cr_request selective_req; }` mapped from `control-<pid>`
(`ShareMemComm::setup`, `src/comm/share_mem.cpp:29` — plain `open` + `ftruncate`
to 2 MB + `mmap MAP_SHARED`; both sides create-if-missing). Selective requests are
copied into `selective_req` *before* the message word is set.

---

# Downstream consumers

`docs/design-memory-address-checkpoint.md` names the **Kubernetes Snapshot Agent's
`BACKEND_GPU_CR_MEMORY_ADDRESSES` backend** as the primary consumer of the selective
API. The interface contract that consumer relies on, per that design doc and
`cr_client.cpp`:

- **Invocation shape (flags).** The agent shells out to exactly
  `cr_client -c -p <pid> -s ptr:size,...` and `cr_client -r -p <pid> -s ...`, with
  an `-i` init once after workload start. Pointers may be hex or decimal;
  ≤ 4096 regions per call. The `-s` list on **restore** is parsed and shipped but
  *ignored* by the worker — `restore_ptr_and_content_selective()` restores whatever
  the dump-file header describes. The dump is the source of truth; the agent must
  therefore manage dump-file contents per snapshot group itself.
- **Dump file layout.** One shared scratch dump file per CUDA process at
  `/mnt/huge-ckpt/<cr_id>` (or `$EXPORT_FILE_PATH/ckpt-<id>.data`), starting with a
  `shared_mem_fs` header — `file_num`, `current_offset`, and up to 4096
  `(ptr, start_offset, size)` entries — followed by packed payload up to
  `current_offset`. The agent copies out only the `current_offset`-limited extent
  into per-group storage and copies it back over the live dump buffer before
  `cr_client -r`. (In v0.2.0 entry sizes are 2 MB-rounded; the unrounded-dumps
  change — commit `7a131b4`, `docs/design-unrounded-dumps.md` — dumps exact sizes.)
  The `pid_map_<pid>` file mapping PID→cr_id ends up **empty on hugetlbfs** (stdio
  writes < page size are lost), so consumers fall back to `/proc/<pid>/maps` to find
  the dump mapping.
- **Exit codes and hang behavior.** `cr_client` exits non-zero only for argument
  parse errors, region-spec errors, or `system()` spawn failures; a *successful*
  selective run exits 0 after `is_finished()`. There is **no timeout**: if the
  workload dies mid-operation (worker-side errors call `exit(-1)` inside the
  handler), `cr_client` polls forever — the agent wraps every call in its own 120 s
  `GPU_CR_OP_TIMEOUT_SEC`. Unresolvable regions (pointer not inside any tracked
  allocation) are *skipped with a warning*, not errors.
- **Workload preconditions the agent enforces:** `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
  (one tensor ≈ one allocation, so block-granularity eviction is safe),
  `CUDA_LAUNCH_BLOCKING=1`, eager mode only, and for vLLM
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the signaled PID owns the addresses.

---

# Gotchas for maintainers

**PID handling.**
- `-p` must be the process that *owns the CUDA context* (the one with vGPU.so
  preloaded and the allocations tracked). vLLM V1 spawns separate worker processes
  by default; either signal the workers (multi flow, `pgrep -f` / `nvidia-smi
  --query-compute-apps=pid`) or force in-process execution
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) for the single/selective flow.
- AMD: `-p` (signal target) and `-m` (CRIU tree root) can differ. CRIU dumps a
  whole *process tree/session* (`-j` = shell job), and restore recreates the same
  PIDs — which is why `cr_client -r` on AMD can still `kill(pid, ...)` the process
  CRIU just resurrected. `serving_vllm_amd.sh` uses `exec` precisely so shell PID ==
  Python PID.
- `multi_cr_client` maps workers by shm-file index 0..N-1, **not** by `-p` order.
  Workers grab CR IDs from an atomic counter in `/mnt/huge-ckpt/control`; if that
  file is stale (previous run), IDs start above 0 and
  `exchange_ipc_export_info()` will fail to open `/mnt/huge-ckpt/0`.
  `launch_multi_gpu.sh` zeroes the counter — do not launch workers by hand without
  doing the same.

**Ordering requirements.**
- **Init before checkpoint.** Both signal handlers lazily call `init_CR()` if
  needed, but `init_CR()` opens files, mmaps ~27 GB of hugepages, and calls CUDA —
  all from inside a signal handler (none of it async-signal-safe). Always run
  `-i` explicitly once, right after model load, so the heavyweight setup happens in
  a controlled moment and checkpoint latency stays predictable.
- **Checkpoint:** teardown → dump → freeze. The dump issues CUDA copies, so it must
  precede the freeze; IPC teardown must precede both.
- **Restore:** unfreeze → data restore → export → exchange → import. Unfreeze must
  be first because a frozen process blocks on every CUDA call; import must follow
  *all* exports plus the CLI-side exchange.
- **Known discrepancy:** single-GPU `cr_client -r` (NVIDIA) sends
  `RESTORE_MSG`+`SIGUSR2` and waits for `FINISH_MSG` *before* running
  `cuda-checkpoint --toggle` — the opposite of `multi_cr_client`'s order and of the
  comment in `vGPU.cpp` ("cuda-checkpoint restore was already called by cr_client
  before this signal"). If the process was genuinely frozen, the handler's
  `cuMemCreate`/`cuMemMap` calls should block until the toggle, i.e., a deadlock.
  It escapes notice partly because `cr_client -c` never checks cuda-checkpoint's
  exit status (see below), so the freeze may have silently failed. Treat
  `multi_cr_client`'s ordering as authoritative; fix `cr_client` before relying on
  its full (non `-b`, non `-s`) restore path.
- Collective phases: always signal all workers before waiting for any
  (`broadcast_and_wait`); with action mode, lock *everyone* before checkpointing
  *anyone*.

**Failure modes when a worker dies mid-phase.**
- No wait loop anywhere has a timeout. Worker dead *before* the signal:
  `broadcast_and_wait` catches the `kill()` failure and exits; `cr_client` just
  polls forever. Worker dies *after* the signal (handler errors call `exit(-1)`):
  both CLIs spin forever at 1 ms intervals. Wrap all invocations in an external
  timeout (the snapshot agent uses 120 s).
- Because `FINISH_MSG == 0` and a fresh control file is zero-filled, "finished" is
  indistinguishable from "never asked". The protocol only works because `send_msg`
  precedes `kill`; if the target never handles the signal (wrong PID, Python-level
  handler swallowed SIGUSR1 — the reason deployment builds move to `SIGRTMAX-8/-7`),
  the word stays non-zero and the CLI hangs.
- `multi_cr_client` action mode attempts `--action unlock` on all workers after a
  failed lock/checkpoint to leave them runnable; toggle mode has no such rollback.
  A partial freeze (some workers toggled, some not) is unrecoverable without manual
  `--get-state` inspection and per-PID toggles.

**Exit-code and small-bug traps in `cr_client.cpp`.**
- `system()` results are checked only for `< 0` (fork failure). A cuda-checkpoint or
  CRIU run that *executes and fails* still lets `cr_client` print
  "Checkpointing done" and exit 0. `multi_cr_client` gets this right
  (`WEXITSTATUS`); mirror that if you touch `cr_client`.
- `int ret;` is read after `if (!buffer_only) ret = system(...)` — with `-b`, `ret`
  is uninitialized (UB; a garbage negative value would abort a run that did
  nothing wrong).
- The getopt string contains `d` but there is no `case 'd':`, and the `dump`
  variable can never become 1 — vestigial.
- `get_cuda_checkpoint_path()` falls back to `$PATH` silently; combined with the
  exit-code blindness above, a missing vendored binary can look like a successful
  checkpoint.

**Environment traps.**
- Multi-GPU workers launched without `NCCL_CUMEM_ENABLE=1` (i.e., not through
  `launch_multi_gpu.sh`) allocate NCCL buffers in a way GPU-CR cannot tear down —
  Phase 1 "succeeds" but the freeze then fails on residual IPC state.
- CUDA Graphs (`enforce_eager` absent) and vLLM custom all-reduce
  (`disable_custom_all_reduce` absent) both break checkpointing in ways that only
  surface at freeze/restore time.
- `SHM_SIZE` is 25 GB per process by default (`-DSHM_SIZE_GB` to change) and the
  hugetlbfs reservation is charged per *file inode* at mmap time — leaked dump
  files from dead workers keep hugepages reserved until deleted.
