# GEP-0003: Verified, Self-Contained Selective Restore (Dump Format v2)

- **Status:** provisional
- **Authors:** Edwin Hernandez (investigation & validation with Claude Code)
- **Creation date:** 2026-08-05
- **Tracking issues:** TBD (upstream GPU-CR), silent-restore-corruption incident 2026-08-04
- **Related:** branch `unrounded-selective-dumps` (orthogonal — reduces dump size, does not address integrity)

## Summary

GPU-CR's selective checkpoint/restore (`cr_client -c/-r -s ptr:size,...`) stores each
checkpoint in a mutable, per-process staging buffer (`/mnt/huge-ckpt/<id>`) that is
**reset and rewritten by every checkpoint** and **trusted unconditionally by every
restore**. External snapshot managers (e.g. the Kubernetes Snapshot Agent) must copy
this buffer out after a checkpoint and reconstitute it byte-for-byte before a restore.
Nothing in the format lets GPU-CR verify that the reconstitution is correct, and the
restore path has no notion of *which* checkpoint it is restoring.

In production this caused **silent cross-checkpoint data transplantation**: restored
GPU regions came back holding payload bytes from a *different* checkpoint group, which
poisoned AdamW optimizer state (`sqrt(negative)`) and destroyed an RL tenant's model
without any operation reporting an error (see [Appendix A](#appendix-a-incident-2026-08-04)).

This proposal makes selective restore **self-describing and self-verifying**:

1. **Dump format v2** — the in-buffer directory gains a magic/version, a monotonic
   `checkpoint_epoch`, a directory checksum, and a per-region payload checksum.
2. **Verified restore** — the restore handler validates the directory and every
   region's payload checksum *before* any HtoD copy, and fails loudly through the
   existing control-file ack channel instead of restoring corrupt bytes.
3. **Restore-from-source** — `cr_client -r` can name an explicit, immutable snapshot
   file; the preloader maps it read-only as the restore source, removing the shared
   mutable staging buffer from the restore path entirely.

## Motivation

The current contract between GPU-CR and any external snapshot manager is implicit and
unverifiable:

- `shared_mem_fs` (src/common.h:77) records `{ptr, start_offset, size}` per region.
  The **payload bytes are authoritative even when they are all-zero** — but nothing in
  the format says so, and nothing detects a manager that assumes otherwise.
- Every checkpoint resets `file_num`/`current_offset` and rewrites the same buffer
  (src/vGPU.cpp:255-301). Two checkpoint groups belonging to the same PID therefore
  occupy the same offsets at different times. Any incomplete reconstitution before a
  restore yields a **plausible-looking hybrid of two checkpoints**.
- The selective-restore handler (src/vGPU.cpp:537+) reads whatever directory is in the
  live buffer and DMAs the recorded ranges to the GPU. It cannot tell a perfectly
  reconstituted snapshot from last operation's residue.

The 2026-08-04 incident hit exactly this seam: the Kubernetes Snapshot Agent's
sparse-file optimizations turned legitimately all-zero payload regions (LoRA `lora_A`
first-step optimizer moments — *exactly zero by construction*) into file holes, and
its restore copy-back left the previous checkpoint's bytes underneath those holes.
GPU-CR restored the hybrid without complaint. The failure surfaced three operations
later as NaN training loss, and in the end-to-end demo as a sampler emitting
deterministic token-0 — past every determinism tripwire.

The immediate agent-side fix (hole-faithful, zero-filling restore copies) closes this
specific bug. This proposal addresses the **class**: GPU-CR should never restore bytes
it cannot attribute to a specific checkpoint, and should detect — not absorb —
manager-side reconstitution errors.

### Goals

- Detect any divergence between the bytes a selective checkpoint produced and the
  bytes a selective restore is about to push to the GPU, and fail the restore with an
  explicit error before the first HtoD copy.
- Give every checkpoint an identity (`checkpoint_epoch`) that external managers can
  record and assert at restore time.
- Allow restores to bypass the shared mutable staging buffer entirely by reading an
  immutable snapshot file directly.
- Preserve wire compatibility with existing external tooling that parses the first 16
  bytes of the dump header (`file_num`, `current_offset`).
- Keep the fast path fast: integrity metadata must add <5% to checkpoint wall time at
  the reference workload (1,008 regions / ~1.5GB unrounded dump on L4).

### Non-Goals

- Fixing the agent's sparse-copy semantics (shipped separately; required regardless
  for v1 compatibility).
- Whole-process (`cuda-checkpoint`) C/R — this KEP covers the selective path only.
- Encryption or authentication of dumps (integrity against *bugs*, not adversaries).
- Sub-block dumps, multi-GPU coordination, or performance work beyond the checksum
  budget above.

## Proposal

### 1. Dump format v2

Extend the directory structures in `src/common.h`. The first 16 bytes keep their v1
layout so existing managers that read `current_offset` at offset 8 (e.g. the agent's
`dumpDataLimit`) continue to work:

```c
#define SHM_FS_MAGIC   0x47435246u   /* "GCRF" */
#define SHM_FS_VERSION 2u

struct shared_mem_file {
    void*    ptr;
    uint64_t start_offset;
    uint64_t size;
    uint64_t payload_xxh64;   /* v2: XXH64 of payload bytes [start_offset, +size) */
};

struct shared_mem_fs {
    uint64_t file_num;         /* offset 0  — unchanged from v1 */
    uint64_t current_offset;   /* offset 8  — unchanged from v1 */
    uint32_t magic;            /* v2 */
    uint32_t version;          /* v2 */
    uint64_t checkpoint_epoch; /* v2: monotonic per-process checkpoint counter */
    uint64_t directory_xxh64;  /* v2: XXH64 over files[0..file_num) */
    struct shared_mem_file files[MAX_FILE_NUM];
};
```

Checkpoint-side changes (selective checkpoint writer, src/vGPU.cpp:240-395):

- Increment a process-global `checkpoint_epoch` (starts at 1) on every selective
  checkpoint and record it in the header.
- Hash each region's payload as it passes through the staging pipeline (the bytes are
  already in cache during the staging→buffer memcpy; hashing there is near-free) and
  record `payload_xxh64` in the directory entry.
- Compute `directory_xxh64` last, after all entries are final.

**The v2 spec makes the payload contract explicit:** every byte in
`[ROUND_UP_2MB(sizeof(shared_mem_fs)), current_offset)` is authoritative checkpoint
data, *including zeros*. External tooling that round-trips the buffer through sparse
files MUST reproduce holes as zeros.

### 2. Verified restore

The selective-restore handler validates before it copies:

1. `magic`/`version` recognized; `file_num <= MAX_FILE_NUM`;
   `current_offset <= SHM_SIZE`; every `{start_offset, size}` within bounds and
   non-overlapping. (Today a garbage directory is followed blindly.)
2. `directory_xxh64` matches.
3. If the restore request carries an `expected_epoch` (see §3), the header's
   `checkpoint_epoch` must equal it. This is what catches "restoring the wrong
   checkpoint's buffer" even when that buffer is internally self-consistent.
4. For each region, recompute XXH64 over the payload and compare with
   `payload_xxh64` — *before* the region's HtoD `memcpyAsync`. Hashing streams
   through the same staging pipeline the restore already uses, so the bytes are read
   once.

On any mismatch the handler **must not** terminate the workload (the current
`exit(-1)` pattern) and must not partially restore. It writes an error code to the
control-file ack word that `cr_client` already polls — new codes
`CR_ERR_BAD_HEADER`, `CR_ERR_EPOCH_MISMATCH`, `CR_ERR_CHECKSUM_MISMATCH(region)` —
and leaves GPU memory as it was (regions are remapped before fill today; validation
of the buffer happens before `remapPhysicalMemory` so no physical state has changed
on the failure path). `cr_client -r` exits non-zero with the code; the agent surfaces
it as `OPERATION_STATUS_FAILED` with the error string.

Escape hatch: `GPUCR_RESTORE_VERIFY=0` skips step 4 (payload hashes) only; header and
epoch checks always run.

### 3. Restore-from-source

Extend the control block (`struct signal_controls`, src/common.h) written by
`cr_client` (coordinator/cr_client.cpp:148-165):

```c
struct selective_cr_request {
    uint32_t num_regions;
    uint64_t expected_epoch;          /* v2: 0 = don't check */
    char     restore_source[512];     /* v2: empty = use live staging buffer (v1) */
    struct selective_cr_region regions[MAX_SELECTIVE_REGIONS];
};
```

New CLI: `cr_client -r -p <pid> -s <spec> --source <path> --epoch <n>`.

When `restore_source` is non-empty, the restore handler `open()`s and `mmap()`s that
file `PROT_READ, MAP_PRIVATE` and uses it as the `shared_mem_fs*` view for the entire
restore (directory + payload), instead of `backend->get_tmp_buf()`. The live staging
buffer is neither read nor written. Combined with §2, a restore is then a pure
function of one immutable file — concurrent checkpoints of *other* groups on the same
PID can no longer contaminate it, by construction.

Deployment note: the source file must be visible inside the workload's mount
namespace at the same path the manager passes. For the Kubernetes Snapshot Agent this
means mounting the snapshot store (e.g. hostPath `/dev/shm/gcr-snapshots`) into
workload pods alongside `/mnt/huge-ckpt`, or having the agent stage the group file to
a per-restore path on an already-shared mount. Until a deployment adopts
restore-from-source, §2 alone already converts the incident's silent corruption into
a loud `CR_ERR_CHECKSUM_MISMATCH`.

### 4. Adjacent hardening (bundled, small)

- Write `pid_map_<pid>` through the existing shared mmap instead of `fprintf`
  (stdio writes fail silently on hugetlbfs, which is why external managers grew a
  `/proc/<pid>/maps` fallback for PID→buffer-id resolution).
- Emit `checkpoint_epoch` and region count on the checkpoint ack so managers can
  record `{group → epoch}` without parsing the buffer.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Checksum cost inflates swap latency | XXH64 streams at >10 GB/s/core; hash during the existing staging memcpys (bytes already hot). Budget: <5% on the 1.5GB reference dump; measured in the perf gate. Escape hatch `GPUCR_RESTORE_VERIFY=0`. |
| Header layout change breaks v1 tooling | First 16 bytes unchanged; v1 readers of `file_num`/`current_offset` unaffected. v2 readers detect v1 buffers by absent magic and fall back to v1 semantics (no verification). |
| Failing restores where v1 silently "succeeded" | That is the point — but roll out verification as warn-only first (Alpha) to size the blast radius before enforcing (Beta). |
| `restore_source` path escapes / wrong file | Handler validates magic/version/epoch before use; manager passes absolute paths it owns; `MAP_PRIVATE` read-only mapping cannot mutate the store. |
| Partial-failure state after a failed verified restore | All validation completes before `remapPhysicalMemory`/HtoD copies begin; failure leaves device memory untouched (still released), and the manager may retry. |

## Test Plan

- **Unit (new, host-only):** XXH64 vectors; directory bounds/overlap validation;
  v1-buffer detection; tamper matrix — flip one payload byte, truncate one region,
  swap two regions' payloads, stale epoch → each must produce its specific error code
  and zero HtoD copies.
- **Integration (GPU):** the incident's regression gate — open-rl
  `scripts/test_trainer_swap_realgrads.py` (two LoRA tenants, real gradients,
  byte-exact restore verification, corrupt-payload source matching) run in the
  vulnerable ordering (`SWAP_IN_FIRST=0`, no warm-up): 1,008/1,008 regions
  byte-identical over ≥4 alternating rounds, both tenants converge.
- **Fault injection (GPU):** run the same gate against a deliberately broken manager
  (agent build with the sparse-copy bug re-introduced): every affected restore must
  fail with `CR_ERR_CHECKSUM_MISMATCH` — the silent-corruption outcome is the test
  failure.
- **Perf gate:** checkpoint/restore wall-time delta vs baseline on the reference
  workload (Qwen2.5-0.5B/rank-16, 1,008 regions; and Qwen3-4B/rank-64, 1,512
  regions); assert <5% regression.

## Graduation Criteria

- **Alpha:** v2 format written behind `GPUCR_DUMP_V2=1`; restore verification runs
  warn-only (logs mismatches, proceeds). Agent records epochs.
- **Beta:** v2 default; verification enforced (fail-closed) with
  `GPUCR_RESTORE_VERIFY=0` escape hatch; error codes plumbed through `cr_client` and
  the Snapshot Agent gRPC status. Fault-injection test in CI.
- **GA:** restore-from-source supported end-to-end in the Kubernetes Snapshot Agent
  and used by default where the snapshot store is mountable; v1 read compatibility
  retained for one release cycle.

## Upgrade / Downgrade / Version Skew

- **New .so + old manager:** v2 buffers keep `file_num`/`current_offset` at their v1
  offsets; old managers copy and reconstitute as before. Verification catches their
  reconstitution bugs (Beta+), which is a behavior change only for managers that were
  already corrupting restores.
- **Old .so + new manager:** manager detects missing magic, records `epoch=0`, skips
  epoch assertions; no verification available (v1 semantics).
- **Downgrade:** v1 readers ignore trailing v2 header fields; payload layout is
  unchanged, so v2 dumps restore under v1 code (without verification).

## Drawbacks

Adds format versioning and ~24 bytes/region of metadata to a hot path that was
format-free; enforcing verification can turn previously "working" (silently lucky)
deployments into failing ones, requiring operator attention. Both are considered
acceptable against a failure mode that destroys model state without any error.

## Alternatives Considered

1. **Agent-side zero-fill only (shipped as the immediate fix).** Correct for the
   known bug, zero GPU-CR changes — but it validates nothing: the next
   reconstitution bug (partial copy, GC race, crashed copy, offset skew) corrupts
   silently again. Kept as a required companion change, not a substitute.
2. **Operation-ordering convention (restore-before-checkpoint at tenant switch;
   `TIMESLICE_SWAP_IN_FIRST`, shipped, validated at 4B scale).** Works because the
   staging residue under any holes is then the same group's own bytes. It is a
   convention at one call site; three-tenant rotations, crash recovery, or any new
   caller can silently violate it. Defense in depth only.
3. **Full non-sparse copies everywhere.** Removes the hole semantics problem without
   format changes, but still restores the wrong checkpoint undetected when identity
   is confused (epoch check is what catches that), and gives up sparse-store savings.
4. **Manager-side checksums (agent hashes the buffer around each op).** Places
   verification in every manager instead of once in GPU-CR, doubles the buffer reads
   (the .so touches the bytes anyway), and cannot verify the final HtoD source at the
   moment of truth.

## Implementation History

- 2026-08-04: Incident: deterministic cross-group restore corruption in a
  two-tenant LoRA trainer; root-caused to sparse-copy hole semantics over the shared
  staging buffer; byte-identical reproduction across nodes; ordering workaround
  validated (0/1008 mismatches × 4 rounds), later default-on and validated at
  Qwen3-4B/rank-64.
- 2026-08-05: This proposal.

## Appendix A: Incident 2026-08-04

Reference workload: Open-RL LoRA trainer, 2 tenants (Qwen2.5-0.5B, rank 16), tenant
switch = `snapshot(A)` then `restore(B)` on the same PID, 1,008 regions ≈ 2GB
(2MB-rounded dumps), Kubernetes Snapshot Agent `hugetlbfs-v5`, L4.

- Tenant-B's first restore returned 126/1,008 regions holding tenant-A's payloads
  (e.g. B's `exp_avg:12` == A's `exp_avg_sq:93` byte-for-byte). Tenant-A's first
  restore had 11 transplanted regions. Reproduced byte-identically across runs and
  nodes.
- Every corrupted region was even-indexed (`lora_A`): after one optimizer step,
  `lora_A` moments are exactly zero (`lora_B` is zero-initialized, so first-step
  `lora_A` gradients vanish) → ≥4MB all-zero runs → agent sparse-copy holes →
  restore copy-back left the co-resident group's fresh bytes underneath.
- Signed `exp_avg` bytes landing in `exp_avg_sq` → `sqrt(negative)` → NaN parameters
  → NaN adapter saved → vLLM greedy decoding over NaN logits → deterministic token-0
  output that passed all determinism tripwires.
- No operation — checkpoint, copy, restore, or training step — reported an error.
