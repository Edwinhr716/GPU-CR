# KEP-0001: Destination-Path Selective Checkpoints

<!--
Follows the Kubernetes Enhancement Proposal (KEP) template structure,
adapted for the GPU-CR repository.
-->

- **Authors**: @Edwinhr716 (with Claude)
- **Status**: provisional
- **Creation date**: 2026-08-05
- **Last updated**: 2026-08-05
- **Related**: `docs/design-memory-address-checkpoint.md`,
  `docs/design-unrounded-dumps.md`,
  llm-d-rl-time-slicing `gcr-backend-memory-allocation` branch (snapshot
  agent `gpu-cr-memory-addresses` backend)

## Summary

Add an optional **destination path** to the selective checkpoint/restore
protocol so a caller can direct each selective dump to (and restore it from)
a caller-chosen file, instead of the single per-PID staging buffer that all
snapshots of a process currently share.

The flag is a pure mechanism: GPU-CR treats the path as an opaque location
and learns nothing about what the snapshot represents. Naming policy,
lifecycle, and garbage collection remain with the caller (e.g. the
llm-d-rl-time-slicing Snapshot Agent, which today encodes tenant identity as
"which directory the agent copies bytes into").

When the flag is absent, behavior is unchanged (per-PID buffer), preserving
full backward compatibility.

## Motivation

Selective checkpointing (`cr_client -c -p <pid> -s ptr:size,...`) writes the
dump into one `ShareMem` buffer per process (`<ctl_dir>/<id>`, resolved
PID→id via `pid_map_<pid>`), sized by the compile-time `SHM_SIZE_GB`
(default 25GiB) and reused by **every** snapshot of that process. Production
use of this path by the llm-d Snapshot Agent for multi-tenant LoRA
swapping (measured on GKE L4 nodes, 2026-08-03/04) surfaced three structural
costs, all traceable to the shared, PID-keyed buffer:

1. **A mandatory copy in the swap path.** Because the next snapshot
   overwrites the buffer, the agent must copy each dump's extents out to a
   durable per-group store and copy them back before restore. The copy
   dominates end-to-end swap latency (measured 80ms–2.5s per operation
   depending on extent size and store backend) and means every parked
   snapshot's bytes exist twice during transitions.

2. **Aliasing hazards between logical snapshots.** Two snapshots of
   different logical groups share one file. A checkpoint(A)-then-restore(B)
   sequence against the same buffer transplanted bytes across groups in
   practice: an adapter's AdamW `exp_avg_sq` received foreign (negative)
   bytes, producing `sqrt(<0) = NaN` weights. The bug was worked around by
   reordering swap operations (restore-before-checkpoint), but the hazard
   class exists as long as dumps share a file.

3. **Reservation and accounting distortions.** The 25GiB buffer is reserved
   on hugetlbfs at `mmap` time regardless of real dump size (observed real
   extents: 0.1–3GB), forcing a 60GiB node hugepage carve-out on 96GB
   machines. Meanwhile the copied groups land in the agent's store (tmpfs),
   charged to the *agent's* memory cgroup — pages that survive its container
   restarts and have OOM-killed it twice in testing. The bytes are owned by
   the workload but billed to a bystander.

### Goals

- Allow a selective checkpoint to be written directly to a caller-specified
  file, and a selective restore to read directly from one.
- Keep GPU-CR policy-free: the path is opaque; no notion of "adapter",
  "tenant", or "group" enters the C++ layer.
- Preserve the existing PID-keyed behavior when no destination is given.
- Make protocol version skew between `cr_client` and `vGPU-*.so` detectable
  (they ship in different container images and update independently).

### Non-Goals

- Lifecycle management of destination files (creation policy, TTL, garbage
  collection) — remains the caller's responsibility.
- Changing full (non-selective) checkpoint/restore.
- Changing dump *content* format. (Data-size reduction is
  `docs/design-unrounded-dumps.md` and composes with this proposal.)
- Multi-GPU orchestration changes (`multi_cr_client`).
- Shrinking `SHM_SIZE_GB` defaults — enabled by this proposal (see Future
  Work) but a separate change.

## Proposal

Add an optional destination to the selective request path:

```
cr_client -c -p <pid> -s ptr:size,...  -o /mnt/huge-ckpt/groups/g-1234
cr_client -r -p <pid> -s ptr:size,...  -o /mnt/huge-ckpt/groups/g-1234
```

`-o <path>`:
- **Checkpoint**: the preloader dumps the selected regions into `<path>`
  (mmap-write), instead of the per-PID buffer.
- **Restore**: the preloader reads region contents from `<path>`.
- **Absent**: current behavior, byte-for-byte.

The caller (not GPU-CR) chooses paths, encodes whatever identity it wants in
them, pre-creates the files, and deletes them when done.

### User Stories

**Story 1 — Multi-tenant LoRA trainer swapping (llm-d Snapshot Agent).**
The agent parks an inactive tenant by checkpointing its adapter + optimizer
regions with `-o <store>/tenant-<uuid>`, and revives it by restoring from
the same path. No agent-side byte copy exists; a tenant switch becomes
signal + GPU DMA. Parked bytes are charged to the file's backing store
(hugetlbfs pages requested by the workload pod, or page-cache/disk for a
disk-backed store), not to the agent.

**Story 2 — vLLM LoRA slot versioning.** The sampler-side manager keeps one
destination file per tenant and overwrites it on each save; restoring an
older session is impossible by construction because the caller controls
naming, exactly as today — but without the double-residency and copy.

**Story 3 — Debugging/forensics.** A developer snapshots a suspect region
set to a named file and diffs two dumps offline, without racing the shared
buffer's next writer.

### Notes/Constraints/Caveats

- **The DMA staging buffers are unaffected.** The 2×1GiB pinned staging
  double-buffer (`<id>-host`) is transient scratch for GPU↔host copies and
  remains per-process. Only the *durable* dump location moves.
- **Caller pre-creates the destination file.** `cr_client` (or the agent)
  performs `open(O_CREAT) + ftruncate(dump_size_upper_bound)` *before*
  signaling, so the signal-handler-context work in the preloader is limited
  to `open + mmap` of an existing file — the same operations `init_CR`
  already performs lazily in handler context today. This also keeps
  hugetlbfs reservation failures (`ENOMEM`) in the caller, where they can be
  reported cleanly, rather than in a signal handler.
- **hugetlbfs constraints apply to the destination** when it lives on the
  hugepage mount: sizes rounded to 2MiB for `ftruncate`, no `write(2)` (all
  access via mmap), reservations charged to the *opening* process's hugetlb
  cgroup. A destination on a regular filesystem trades speed for
  spill-to-disk capacity; both are valid caller choices.
- **Path length** is bounded by the control-channel field (256 bytes,
  below).

### Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| ABI skew: new `cr_client` + old `.so` (or vice versa) misread the control struct | Version byte in the extended struct (below). Old `.so` ignores unknown message variants and fails the op with a logged error; new `cr_client` refuses `-o` when the `.so` doesn't ack the capability. |
| File creation/reservation failure at checkpoint time | Caller-side pre-create (see above): failure surfaces in `cr_client` stderr/exit code before any signal is sent; workload state untouched. |
| Path traversal / writing outside the intended store | GPU-CR intentionally does **not** police paths (mechanism, not policy); the *caller* must validate, as the llm-d agent already does for group names. Documented explicitly. |
| Per-operation `mmap/munmap` churn vs. today's long-lived mapping | Dumps map only the extent actually written (with unrounded dumps: the real data size), not 25GiB; measured mmap cost at these sizes is microseconds-to-milliseconds, far below the copy it replaces. |
| Orphaned destination files pin hugetlbfs RAM | Unchanged ownership model: the caller that names files sweeps them (the llm-d agent already runs dead-PID/TTL GC and would extend it to its own store). |

## Design Details

### Control-channel protocol

`struct selective_cr_request` (in `src/common.h`) gains a versioned
extension:

```c
#define SELECTIVE_CR_PROTO_V2      2
#define SELECTIVE_CR_MAX_PATH    256

struct selective_cr_request {
    uint32_t num_regions;
    struct selective_cr_region regions[MAX_SELECTIVE_REGIONS];
    /* --- v2 extension (appended; offset invisible to v1 readers) --- */
    uint32_t proto_version;             /* 0 (v1 zero-init) or 2 */
    char     dest_path[SELECTIVE_CR_MAX_PATH]; /* empty = per-PID buffer */
};
```

- The control file is `ftruncate`d to one hugepage (2MiB) today, so the
  appended fields fit without resizing.
- A v1 `.so` never reads past `regions[]`; a v2 `.so` treats
  `proto_version < 2 || dest_path[0] == '\0'` as "use the per-PID buffer".
- Zero-initialization of the control mapping (already the case for fresh
  hugetlbfs pages) makes the v1→v2 default safe.

Message codes `SELECTIVE_CKPT_MSG` / `SELECTIVE_RESTORE_MSG` are unchanged;
the destination is data, not a new verb.

### cr_client

- New `-o <path>` flag for `-c` and `-r` with `-s`.
- On `-o`: validate length; `open(O_CREAT|O_RDWR)` + `ftruncate` the
  destination to the region total (2MiB-rounded on hugetlbfs) before
  writing the control struct and signaling.
- Populate `proto_version = 2` and `dest_path`; poll for completion as
  today.

### vGPU preloader (`src/vGPU.cpp`)

- `ckpt_selective()`: if `dest_path` is set, `open + mmap` the destination
  and lay out the dump there (same on-disk format as today: `shared_mem_fs`
  header + extents; with the unrounded-dumps change, extents are
  `alloc_size`); otherwise use `backend->get_tmp_buf()` as today.
  Physical-release semantics are unchanged.
- `restore_ptr_and_content_selective()`: symmetric read side.
- `munmap` the destination at operation end; the per-PID buffer's lifetime
  is untouched.

### Consumer migration (informative, out of tree)

The llm-d Snapshot Agent's `gpu-cr-memory-addresses` backend replaces its
copy phases with path selection: `Snapshot` = pre-create + `cr_client -c -o
<store>/<group>`; `Restore` = `cr_client -r -o <store>/<group>`. Its GC
extends to the store directory. Its memory cgroup requirement collapses from
"sum of all parked groups" to megabytes.

### Test Plan

- **Unit**: flag parsing; struct version defaulting (zeroed v1 struct read
  by v2 code and vice versa); path-length bounds.
- **Integration** (existing GKE rig from the llm-d demo):
  - `test_lora_swap_max1` variant using `-o`: hijack-restore determinism
    (temp-0 outputs byte-stable across A/B swaps).
  - Trainer gate variant: bitwise tensor comparison + optim-step-after-
    restore, with per-tenant destination files; assert the aliasing
    reproducer (checkpoint A, checkpoint B, restore A) yields A's bytes —
    the case the shared buffer fails without ordering workarounds.
  - Skew matrix: {v1,v2} `cr_client` × {v1,v2} `.so`, expecting graceful
    refusal in mixed pairs and unchanged v1 behavior.
- **Perf**: swap-op latency and bytes-moved vs. the copy-based agent on the
  same workload (baseline numbers exist from the 2026-08-04 runs).

### Graduation Criteria

- **Alpha**: flag exists, off-path unless `-o` passed; integration tests
  green on the llm-d rig.
- **Beta**: llm-d agent migrated behind an env toggle; skew matrix in CI;
  aliasing reproducer test permanent.
- **GA**: agent copy path removed; `SHM_SIZE_GB` default revisited (see
  below).

## Drawbacks

- Extends a shared-memory ABI that two independently-shipped binaries must
  agree on; version skew becomes a real (if detectable) failure mode.
- Identity, while opaque, now *transits* the C++ layer as a path — the
  strict "GPU-CR knows only PIDs" invariant is relaxed to "GPU-CR knows
  PIDs and caller-opaque destinations".
- Per-operation file open/mmap in the preloader adds code to a
  signal-handler context that is already delicate.
- Does not reduce total parked-state memory: N parked snapshots still cost
  N × extent bytes; this proposal only removes double-residency, the copy
  latency, and misattributed accounting.

## Alternatives

1. **Status quo (agent-side copy).** Works today; costs are the measured
   copy latency, double residency, aliasing hazard (mitigated only by
   operation ordering), and agent-cgroup billing.
2. **Group/tag flag instead of a path** (`-g adapter-X`, GPU-CR derives the
   filename). Rejected: leaks naming policy and tenant semantics into
   GPU-CR, requires GPU-CR-side namespace management, and still needs the
   same struct extension. The path variant is strictly more general.
3. **Shrink `SHM_SIZE_GB` and keep the copy.** Fixes only the reservation
   distortion; copy, aliasing, and accounting costs remain. Complementary,
   not alternative (see Future Work).
4. **Per-snapshot rotation of the PID buffer** (N buffers per process,
   round-robin). Avoids the protocol change but multiplies reserved
   buffers, keeps PID-keying (callers still can't address a specific
   snapshot), and leaves the accounting misattribution.

## Future Work

- Right-size `SHM_SIZE_GB` (compile-time today; runtime-configurable
  ideally) once destinations carry durable dumps — enables shrinking node
  hugepage reservations from 60GiB to ~20GiB on 2×L4 nodes.
- Compose with unrounded dumps (`docs/design-unrounded-dumps.md`):
  destination files sized to real state (e.g. 93MB instead of 2GB per
  rank-16 tenant) make dense multi-tenant parking practical on tmpfs.
- Optional direct-to-disk destinations as a documented spill tier.

## Implementation History

- 2026-08-03/04 — llm-d × Open-RL multi-tenant LoRA demo quantifies the
  copy path (swap ops 388ms–2.5s copy-dominated), the aliasing bug
  (cross-group byte transplant → NaN optimizer state; fixed by operation
  reordering), and the accounting failure mode (agent OOM from parked
  groups).
- 2026-08-04 — unrounded-dumps validated (22–58× dump-size reduction),
  making destination files small enough for dense parking.
- 2026-08-05 — this proposal.
