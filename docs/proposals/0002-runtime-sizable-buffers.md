# KEP-0002: Runtime-Sizable Checkpoint Buffers

<!--
Follows the Kubernetes Enhancement Proposal (KEP) template structure,
adapted for the GPU-CR repository.
-->

- **Authors**: @Edwinhr716 (with Claude)
- **Status**: provisional
- **Creation date**: 2026-08-05
- **Last updated**: 2026-08-05
- **Related**: `docs/proposals/0001-selective-checkpoint-destination-path.md`,
  `docs/design-unrounded-dumps.md`

## Summary

Make the per-process checkpoint buffer sizes runtime-configurable instead of
compile-time constants. Today the dump buffer is `SHM_SIZE_GB` (default
25GiB) and the DMA staging buffers are `STAGING_BUF_SIZE × STAGING_BUF_NUM`
(1GiB × 2), all `#define`s: changing a deployment's memory footprint
requires rebuilding `vGPU-*.so`. This proposal reads sizes from the
environment once at library load, validates selective operations against the
actual buffer size with a clean failure, and keeps today's values as
defaults.

Notably, deployment manifests in the wild **already set** `GPUCR_SHM_GB` and
`GPUCR_STAGING_MB` — env vars that no code has ever read. The configuration
contract exists by usage; this proposal implements it (with a migration
caveat, see Drawbacks).

## Motivation

- **Rebuilds for a number.** The buffer size has no cross-binary ABI
  dependency: only the preloader maps the dump buffer (`cr_client` maps just
  the 2MiB control file, and consumers copy by file size). It is
  compile-time purely because it is a macro.
- **The reservation drives node economics.** Each CUDA process reserves
  `SHM_SIZE + 2×STAGING` on hugetlbfs at `mmap` time regardless of real dump
  size. Measured real extents in the llm-d multi-tenant LoRA deployment are
  0.1–3GB against a 25GiB reservation; two GPU workloads force a 60GiB
  hugepage carve-out on a 96GB node, leaving ~27GiB of general-purpose RAM.
  Right-sizing to 8GiB per process shrinks the carve-out to ~22GiB and
  returns ~38GiB to the node.
- **One image, many roles.** The same workload image serves trainers,
  samplers, and batch jobs with very different footprints; per-pod env is
  the natural knob, per-role image builds are not.
- **The status quo actively misleads.** `GPUCR_SHM_GB=60` sits in deployed
  YAML today and silently does nothing. Operators reading those manifests
  reasonably conclude the knob exists. A configuration surface that parses
  as tunable but is inert is worse than no surface.

### Goals

- Size the dump buffer and staging buffers from environment variables read
  once at library load; defaults identical to today (25GiB / 1GiB×2).
- Validate each selective checkpoint's region total against the actual
  buffer size *before* touching GPU state; fail the operation cleanly.
- Detect legacy no-op variable names and warn loudly (see migration
  discussion).
- Zero protocol/ABI change: `cr_client` and the `.so` need no coordination.

### Non-Goals

- Growing or shrinking a buffer after initialization (hugetlbfs offers no
  realloc; remapping live shared state is not worth it).
- Automatic sizing from observed workload behavior.
- Changing node-level hugepage provisioning itself (operator action, see
  Drawbacks).
- Destination-path dumps (KEP-0001; composes, below).

## Proposal

Two new canonical variables, read in the library constructor (not in the
lazy signal-handler init — see Design Details):

| Variable | Meaning | Default |
| --- | --- | --- |
| `GPU_CR_SHM_GB` | dump buffer size, GiB, 2MiB-aligned | 25 |
| `GPU_CR_STAGING_MB` | size of each of the two staging buffers, MiB | 1024 |

Semantics:

- Unset / unparsable / out of bounds (below 1GiB dump, 128MiB staging;
  above 25GiB / 1GiB): fall back to default with a logged warning — never
  fail load, never silently clamp without a log line.
- Legacy names `GPUCR_SHM_GB` / `GPUCR_STAGING_MB`, if present, are **not
  honored** but produce a prominent warning naming the canonical variable
  (rationale in Drawbacks/Alternatives).
- At `SELECTIVE_CKPT_MSG`: if `Σ region sizes + header` exceeds the dump
  buffer, log the shortfall and fail the operation without dumping or
  releasing any region.

### User Stories

**Story 1 — Right-sized nodes.** The llm-d LoRA deployment sets
`GPU_CR_SHM_GB=8` on trainer and sampler pods, recreates the node pool with
a ~22GiB hugepage config instead of 60GiB, and reclaims ~38GiB of RAM per
node for the agent's snapshot store, job memory, and additional parked
tenants — no image rebuild.

**Story 2 — Mixed fleet, one image.** A full-model checkpoint job on the
same cluster keeps the 25GiB default; LoRA pods run at 8GiB; both pull the
same `vGPU-NVIDIA.so`.

**Story 3 — CI on small GPUs.** Integration tests run with
`GPU_CR_SHM_GB=2` on nodes with a 6GiB hugepage pool, instead of skipping
because the default reservation cannot fit.

### Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Undersized buffer discovered only when a big checkpoint arrives | Pre-op validation fails the operation before any GPU region is released; nothing is dumped partially. Logged with requested vs. available bytes. |
| Env read in async-signal context | Sizes are read and cached in the library constructor (`Library loaded!` path), not in `init_CR`'s signal-handler-lazy path; the handler only consumes cached values. `setenv` after load is documented as ineffective. |
| Operator sets env larger than the pod's hugepage request or the node pool | `mmap` fails at init exactly as today with pool exhaustion; the error message is extended to print the configured size and the likely mismatch. This class of misconfiguration is inherent to the design (see Drawbacks #2). |

## Design Details

- `src/common.h`: `SHM_SIZE` / `STAGING_BUF_SIZE` become
  `gpu_cr_config().shm_size` / `.staging_size` — a small struct populated
  once by a constructor-attribute function that parses, bounds-checks, and
  logs the effective values.
- `src/backend/mmap_backend.cpp` (`ShareMem::setup`): use the config struct
  for `ftruncate`/`mmap` sizes (both file and hugetlbfs branches).
- `src/vGPU.cpp` (`ckpt_selective`): compute the dump's required bytes from
  the region list before the first copy; on overflow, log and signal
  operation failure; the selective allocator additionally hard-bounds by
  the configured size (defense in depth).
- Docs: README hugepage-sizing math parameterized by the variables;
  user-guide examples updated (they currently show the inert legacy names).

### Failure signaling caveat

The control channel has no error field — completion is a bare `FINISH_MSG`.
A cleanly-failed operation is therefore visible to `cr_client` today only
as "finished" (with the error on the workload's stderr) or, if we withhold
`FINISH_MSG`, as a poll timeout. Interim choice: write `FINISH_MSG`, log
loudly on both the preloader side and (via missing dump growth) rely on the
consumer's own verification. The clean fix is an error/status byte in the
control struct — which KEP-0001's versioned extension already introduces;
implementations landing together should share it. Until then, this is an
acknowledged diagnostic gap (Drawbacks #4).

### Test Plan

- **Unit**: parse/bounds/default matrix for both variables; legacy-name
  warning emission; config-struct single-initialization.
- **Integration** (llm-d rig): 8GiB env on a 22GiB-pool node pool — full
  trainer gate + sampler self-test green; oversized-request test (region
  total > buffer) fails the op and leaves training functional; env larger
  than pool fails at init with the enriched message.
- **Regression**: no env set → byte-identical behavior to current builds,
  verified against the existing gate suite.

### Graduation Criteria

- **Alpha**: variables implemented, defaults unchanged, gates green with
  and without env.
- **Beta**: llm-d deployment migrated to `GPU_CR_SHM_GB=8` + resized pool;
  README math updated; legacy-name warning in place for one release.
- **GA**: user guide shows only canonical names; error/status byte shared
  with KEP-0001 replaces the failure-signaling caveat.

## Drawbacks

Stated fully, per the intent of this proposal:

1. **A new runtime footgun replaces a build-time inconvenience.** Today an
   oversized checkpoint is impossible by construction (25GiB fits any
   plausible selective dump); after this change, an operator can configure
   a buffer smaller than a workload's real need and discover it only when
   a large checkpoint fails at runtime — potentially long after the config
   change that caused it, in a different team's workload.
2. **Three-way consistency becomes the operator's job, by hand.** The env
   value, the pod's `hugepages-2Mi` request, and the node pool's hugepage
   config must agree, and no layer can validate the others: the scheduler
   cannot see the env var's implication, and the preloader cannot see the
   pool. Today at least one side of that triangle is a fixed constant.
   Misconfiguration modes: env < request (silent hugepage waste — the exact
   pathology this proposal exists to fix, reintroduced piecemeal), env >
   request (init `ENOMEM`), pool < Σ requests (scheduling or init failure
   depending on ordering).
3. **Retroactive activation hazard is why legacy names are rejected —
   at the cost of abandoning the de-facto contract.** Honoring
   `GPUCR_SHM_GB` would make years of deployed manifests suddenly
   meaningful: the llm-d deployment itself ships `GPUCR_SHM_GB=60`, which
   on a 60GiB pool would turn a working deployment into an init failure
   (60 + staging > pool) on the first image update. Using new names avoids
   that, but means the variables operators already "know" remain inert
   (now with a warning) — a genuinely awkward compromise either way.
4. **Failure diagnostics are weak until the control-struct gains a status
   field.** A rejected oversized checkpoint surfaces on the workload's
   stderr, not through `cr_client`'s exit code (see Design Details). An
   operator watching only the agent sees a "successful" operation with a
   missing dump.
5. **It saves nothing by itself.** Defaults are unchanged, and realizing
   the benefit requires coordinated action — pod env, pod hugepage
   requests, and node-pool recreation (~15 min disruption per pool). The
   proposal creates the possibility of right-sizing, not the fact of it.
6. **Documentation debt multiplies.** Every piece of sizing guidance
   (README multi-GPU math, user guides, deployment examples) becomes
   parameterized rather than concrete, and stale copies of the old
   constants in downstream docs will mislead until found.

## Alternatives

1. **Status quo: compile-time, one image per footprint.** Drawbacks: image
   matrix growth per (role × size), registry/rebuild overhead per change,
   defeats single-image multi-role deployments, and leaves the misleading
   inert env vars in the wild.
2. **Honor the legacy variable names.** Respects the documented-by-usage
   contract and requires no manifest edits for intentional users.
   Drawbacks: the retroactive-activation hazard above is disqualifying —
   it converts existing, working-by-accident deployments into failing ones
   on upgrade, with the failure appearing unrelated to any operator action.
3. **Size communicated by `cr_client` at INIT time.** Centralizes config at
   the agent. Drawbacks: it is an ABI change (control-struct field →
   version skew class of KEP-0001) for a value that has no cross-binary
   dependency; and the preloader may need the buffer before any client has
   ever contacted it.
4. **Config file on the shared checkpoint mount.** Works without env
   plumbing. Drawbacks: introduces a file contract with ordering/race
   semantics at init, couples configuration to a hostPath that exists for
   data, and is harder to express in pod specs than env.
5. **Grow-on-demand buffers.** Eliminates sizing entirely. Drawbacks:
   hugetlbfs has no realloc; growing means remap-and-copy of a live shared
   mapping coordinated across the signal protocol — complexity wildly out
   of proportion to setting a number correctly.

## Future Work

- Share the control-struct status/version extension with KEP-0001 so
  undersize rejections surface in `cr_client` exit codes.
- With KEP-0001 landed, the per-PID dump buffer serves only the legacy
  path; its default can eventually drop far below 25GiB, making this
  proposal's careful sizing mostly moot — the desired end state.
- A small sizing helper (`cr_client --plan <regions-file>`) that prints
  required buffer/pool numbers, closing part of the three-way consistency
  gap operationally.

## Implementation History

- 2026-08-03 — Phase 0 of the llm-d demo discovers `GPUCR_SHM_GB` /
  `GPUCR_STAGING_MB` / `GPUCR_NUM_THREADS` are read by nothing (source grep
  + `strings` on the shipped binary), while deployed manifests set them.
- 2026-08-04 — 60GiB node carve-out identified as the binding constraint on
  agent snapshot-store capacity and job scheduling during the multi-tenant
  wall-run design; ~38GiB reclaimable at `SHM=8GiB` per process.
- 2026-08-05 — this proposal.
