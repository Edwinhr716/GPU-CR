# KEP-0004: Caching-Allocator-Compatible Selective Checkpoint (Removing `PYTORCH_NO_CUDA_MEMORY_CACHING=1`)

<!--
Follows the Kubernetes Enhancement Proposal (KEP) template structure,
adapted for the GPU-CR repository.
-->

- **Authors**: @Edwinhr716 (with Claude)
- **Status**: provisional
- **Creation date**: 2026-08-05
- **Last updated**: 2026-08-05
- **Related**: `docs/design-memory-address-checkpoint.md` (assumption §6.1),
  `docs/future-improvements.md` (items 2, 3, 7, 9),
  `docs/proposals/0001-selective-checkpoint-destination-path.md` (control-struct
  versioning), `docs/proposals/0002-runtime-sizable-buffers.md` (status byte)

## Summary

Selective checkpoint/restore today hard-requires
`PYTORCH_NO_CUDA_MEMORY_CACHING=1` on every hooked workload. This proposal
removes that requirement — not by teaching GPU-CR to operate on fractions of a
caching-allocator segment (shown below to be unfixable from GPU-CR's position
in the stack), but by **making tenant state segment-exclusive** with
per-tenant `torch.cuda.MemPool`s, so that GPU-CR's existing whole-block
contract becomes correct again with caching enabled everywhere.

Two phases:

- **Phase A** — per-tenant MemPool segregation. Consumer-side pool plumbing
  plus a small set of GPU-CR hardening changes (a real `cudaFree`-while-parked
  fix, strict region validation, allocator-config guards). No protocol change.
- **Phase B (optional)** — first-class tenant arenas: a thin
  `CUDAPluggableAllocator` backend that tags segments with a tenant ID inside
  GPU-CR, and a `cr_client -t <tag>` operation that checkpoints by tag. This
  eliminates the address-discovery contract entirely.

Phase A also collapses two adjacent costs measured in the field: the
~1500-blocks-per-tenant driver-call floor (~70–110 ms per release pass) drops
to a handful of segments, and the 2 MB-per-tensor resident inflation
(1.47 GB tenant → 3.8 GB resident) disappears because segments are large.

## Motivation

`PYTORCH_NO_CUDA_MEMORY_CACHING=1` exists because of two facts, and any
honest removal plan has to address both:

1. **Co-tenancy.** GPU-CR tracks, dumps, releases, and remaps whole
   `cudaMalloc` allocations — one VMM handle per block
   (`src/GPUs/NVIDIA/nv.cpp`, hook at `cudaMalloc`; `releasePhysicalMemory` /
   `remapPhysicalMemory` operate on the block's single handle). With the
   caching allocator on, one `cudaMalloc` segment holds many tensors with
   different owners. Checkpointing "one tensor" dumps its neighbors;
   releasing its segment corrupts them.
2. **The reuse hazard.** Worse and less obvious: while a tenant is parked,
   its VA range is reserved but **unbacked**. The caching allocator does not
   know this — freed blocks inside that segment remain on its free lists, and
   the next allocation from any thread can be placed into unbacked memory.
   The first kernel write is an illegal memory access at best, silent
   corruption at worst. GPU-CR sits *below* the allocator (an LD_PRELOAD on
   `cudaMalloc`) and has no interposition point on block placement, which
   happens in user-space PyTorch code with no CUDA call.

Fact 2 is the load-bearing one: **no purely GPU-CR-side change can remove
the requirement safely.** Sub-block release granularity, driver-API
interception, chunked handles — all still leave the allocator free to hand
parked bytes to a live tensor. The allocator layer must cooperate. The
cheapest sufficient form of cooperation is segment exclusivity: if every
segment a tenant touches belongs to that tenant alone, whole-segment
eviction is correct (fact 1 solved), and a parked tenant's segments receive
no new allocations because a parked tenant, by definition, isn't allocating
and its pool serves nobody else (fact 2 solved).

PyTorch has shipped exactly this primitive since 2.5:
`torch.cuda.MemPool` + `torch.cuda.use_mem_pool` create private
caching-allocator pools whose segments are never shared with the global pool
or other pools, while keeping caching semantics (fast reuse, stream safety)
inside each pool. The pool's segments are obtained through the same
`cudaMalloc` path GPU-CR already hooks.

What the current requirement costs, per `future-improvements.md` §7:
every allocation in the process pays the no-caching tax (an ~allocation-rate
slowdown on all tensors, not just tenant state), every tensor pays the 2 MB
VMM floor (measured 2.6× resident inflation at 4B/rank-64), per-tensor
allocation pushes region counts toward the 4096 hard limit (1502 regions
already), and park/restore pays ~1500× the driver-call floor.

Measured at fleet level (tenant-ceiling campaign, 2026-08-06, Qwen3-4B/
rank-48, one L4): a resident multi-tenant trainer survives **10 tenants**
under the stock allocator but only **2–3** with this environment applied —
the requirement consumes most of the capacity that selective C/R then
reclaims, and roughly half of the headline "VRAM freed per parked tenant"
(3.2 GB vs ~1.5 GB packed) is inflation the requirement itself created.

### Goals

- Selective checkpoint/restore correct with the PyTorch caching allocator
  **enabled process-wide** (no `PYTORCH_NO_CUDA_MEMORY_CACHING`).
- Whole-block contract preserved: dump units are caching-allocator segments
  that are tenant-exclusive by construction.
- Strict validation so that a region list that violates the contract
  (interior pointer, unknown address, stale segment) fails loudly instead of
  silently evicting a neighbor.
- Fix the existing `cudaFree`-of-a-parked-pointer VA leak, which graduates
  from latent to likely once `empty_cache()` can free segments.
- Phase B: checkpoint-by-tenant-tag, removing address discovery from the
  agent contract.

### Non-Goals

- Removing `CUDA_LAUNCH_BLOCKING=1` (separate root-cause effort,
  `future-improvements.md` §7 bullet 1).
- CUDA graphs / `torch.compile` support (the park/remap cycle remains
  invisible to captured graphs).
- `expandable_segments:True` or `backend:cudaMallocAsync` interop — both
  bypass the hooked `cudaMalloc` symbol; Phase A explicitly detects and
  rejects them (see Design Details).
- AMD/ROCm parity in the first cut (MemPool-on-ROCm maturity unverified;
  NVIDIA-first, mirroring the selective feature itself).
- Multi-GPU/TP selective C/R (`future-improvements.md` §10).

## Proposal

### Phase A — tenant-exclusive pools, hardened core

**Workload contract (consumer-side, replaces the env-var contract).** For
each tenant, the trainer/sampler creates one long-lived
`torch.cuda.MemPool` and enters `torch.cuda.use_mem_pool(pool)` around
exactly the allocations that constitute parkable state: adapter weight
loading and the first optimizer step (AdamW state materializes lazily
there). Gradients and activations stay in the global caching pool — they are
transient and never parked. The pool object is kept alive for the tenant's
whole residency, parked spans included, so `empty_cache()` cannot release
parked segments out from under the dump header.

**Address discovery.** The agent enumerates the pool's segments (base
address + size) via `torch.cuda.memory_snapshot()` — segments record their
owning pool in current PyTorch; where they don't, diffing snapshots around
pool creation is the fallback — and passes segment bases as today's
`-s ptr:size` list. Segment bases are true allocation bases, so the existing
whole-block semantics apply unmodified. Segment count per tenant is a
handful, not ~1500, retiring the region-count headroom concern (§9).

**GPU-CR changes (all in the preloader, no protocol change):**

1. **`cudaFree` of a parked pointer** (`nv.cpp` hook): today, when
   `global_handle_map` has no handle (i.e., the block is parked), the hook
   skips `cuMemUnmap`/`cuMemRelease` *and* `cuMemAddressFree`, leaking the VA
   reservation while erasing the tracking entry. Fix: always release the VA
   reservation; if the freed pointer appears in the current dump header, log
   prominently (a restore of that dump will now fail cleanly at
   `remapPhysicalMemory`'s unknown-pointer check rather than remapping into
   a recycled VA).
2. **Strict region validation** (implements `future-improvements.md` §2 for
   this path): a region must name an exact tracked base (or, at minimum, be
   fully contained with `size ≤ alloc_size` — configurable strictness);
   unknown and interior pointers fail the *operation* before any dump or
   release, with per-region verdicts logged. Failure surfaces through the
   control-struct status byte shared with KEP-0001/0002 once that lands;
   until then, stderr + refused dump (same interim caveat as KEP-0002).
3. **Allocator-config guard**: at library load, inspect
   `PYTORCH_CUDA_ALLOC_CONF`; warn-and-refuse-selective if
   `expandable_segments:True` or `backend:cudaMallocAsync` is set (those
   allocations never traverse the hook, so tenant state would be invisible —
   today this is a silent"address not in any allocated block" skip at
   checkpoint time, i.e., silent non-protection).
4. **Gate tests**: caching-on integration gate — two tenant pools + shared
   base model; park A, train B, restore A, write-after-restore, then
   byte-compare B's state against a no-park control run (neighbor-integrity
   check); interior-pointer and stale-address requests must fail cleanly.

Dump amplification note: the dump unit becomes the segment, so a dump
carries the pool's cached-but-free slack. This slack is bounded by the
tenant's peak working set and replaces today's per-tensor 2 MB rounding
inflation — for LoRA-scale tenants a strict improvement (measured today:
48× at opt-125m/rank-16 from rounding alone, pre-unrounded-dumps).

### Phase B — first-class tenant arenas (optional, follows A)

A ~small pluggable backend: `MemPool` accepts a `CUDAPluggableAllocator`;
GPU-CR ships `libgpucr_pool.so` exporting `gpucr_pool_malloc/free` plus a
Python shim that creates one tagged allocator per tenant
(`gpucr.tenant_pool("tenant-a")`). Segment allocations then arrive through
an explicit GPU-CR entry point that records `tenant_tag → blocks` in the
preloader. `cr_client` gains `-t <tag>` (checkpoint/restore everything
tagged), carried in a versioned control-struct extension per KEP-0001's
scheme. The agent no longer discovers, transports, or risks staleness of
addresses; the stale-address silent-eviction hazard (design doc §6.3)
disappears for tagged tenants.

### User Stories

**Story 1 — full-speed multi-tenant trainer.** The llm-d LoRA trainer drops
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`, wraps `load_adapter` and first
`optim.step()` per tenant in `use_mem_pool`, and passes segment bases to the
agent. Base-model and activation allocations run at native caching-allocator
speed; a 1.47 GB tenant occupies ~1.5 GB resident instead of 3.8 GB; park
issues ~6 release calls instead of ~1500.

**Story 2 — clean failure instead of corruption.** A tenant reallocates
after `delete_adapter`/`load_adapter`; the agent's cached segment list is
stale. Under strict validation the checkpoint is refused with per-region
verdicts, instead of today's silent eviction of whichever block now occupies
that address.

**Story 3 (Phase B) — no address plumbing.** The agent parks tenant-a with
`cr_client -c -p PID -t tenant-a`. No `/proc/<pid>/maps` scraping, no
`memory_snapshot()` calls, no 4096-entry lists.

### Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A tenant allocation escapes the `use_mem_pool` scope (e.g., a lazy buffer materialized later) and lands in the global pool | It is then simply not parked — strict validation plus a consumer-side "pool bytes ≈ expected tenant bytes" sanity check make the gap visible; today's equivalent (address list missing a tensor) is the same failure but silent. |
| `MemPool` API surface shifts across PyTorch releases (2.5→2.7 changed semantics of pool release/`empty_cache` interaction) | Pin and CI-test the supported PyTorch range; the GPU-CR side is version-independent (it sees only `cudaMalloc`/`cudaFree`). |
| `empty_cache()` frees a live (unparked) tenant segment between discovery and checkpoint | Same staleness class as today's contract (design doc §6.3); strict validation converts it from silent eviction to a refused operation; Phase B removes the window entirely. |
| Signal arrives while a hook thread holds `gpu_mem_mutex` mid-`cudaMalloc` → handler deadlock | Pre-existing exposure; caching-on *reduces* it (hook traffic drops from every-tensor to new-segments-only). A handler-side deadline (`future-improvements.md` §2) remains the structural fix. |
| ROCm behavior unknown | NVIDIA-only gate for Phase A; AMD tracked separately. |

## Effort Estimate

One engineer, familiar with the codebase, gates run on the llm-d rig.

| Work item | Where | Estimate |
| --- | --- | --- |
| `cudaFree`-while-parked VA fix + unit tests | `src/GPUs/NVIDIA/nv.cpp` | 1–2 days |
| Strict region validation + per-region verdicts (interim stderr signaling) | `src/vGPU.cpp`, `src/common.h` | 3–5 days |
| Allocator-config guard + honest logging | `src/vGPU.cpp` init path | 0.5 day |
| Caching-on integration gate (2 tenants, neighbor-integrity, failure cases) | test harness | 3–4 days |
| Docs: user guide, design-doc assumption §6.1 rewrite, README | docs | 1–2 days |
| **GPU-CR subtotal (Phase A)** | | **~2 weeks** |
| Trainer integration: per-tenant pools, discovery via `memory_snapshot()`, gate reruns | consumer repo | 1–2 weeks |
| vLLM sampler integration: pool-scoped `load_lora`; vLLM controls its own allocation internals, so this carries real unknowns | consumer repo | 1–3 weeks (spike first) |
| **Phase A total** | | **~4–6 engineer-weeks** end-to-end, of which ~2 in this repo |
| Phase B: pluggable backend `.so` + Python shim + tag registry + `-t` protocol (versioned control struct) + gates | this repo + thin consumer change | 2–3 weeks |

The vLLM line is the schedule risk; the trainer path alone (Phase A minus
vLLM) is the recommended first milestone and is sufficient to delete the env
var for the training workload.

## Complexity Impact

**Phase A, this repo: small and mostly subtractive in character.**
Net ≈ +300–600 LOC (validation, guards, the `cudaFree` fix, tests). No new
subsystem, no new process, no ABI change (the status byte is inherited from
KEP-0001/0002, not introduced here). Two existing latent hazards
(VA-reservation leak, silent whole-block eviction) become handled paths, so
part of the diff is hardening the code should have had anyway.

**Phase A, conceptual:** the complexity does not vanish — it moves from an
env var ("caching off, everywhere, forever") to a **pool-discipline
contract** ("tenant state allocates inside its pool; the pool outlives the
park"). That contract lives in consumer code and docs, is violated silently
at allocation time, and only detected at checkpoint time by validation. This
is the proposal's real cost, and it is judged worth it because the env var's
cost is paid by every allocation on every workload, while the contract's
cost is paid only at tenant-state allocation sites (a handful per consumer).

**Phase B, this repo: moderate.** A new deliverable (`libgpucr_pool.so` +
shim, ~500–800 LOC), a tag registry in the preloader, and one versioned
control-struct extension. In exchange it deletes the largest piece of
*system* complexity the feature has: the externally-discovered,
staleness-prone address list and its 4096-entry plumbing through agent,
control file, and header.

## Drawbacks

1. **The consumer must change.** Today any PyTorch workload works by setting
   two env vars; after this, correct operation requires code-level pool
   plumbing per tenant-state allocation site. Frameworks that hide
   allocation (vLLM internals, DeepSpeed) need per-framework integration
   work, and until done, those consumers stay on the env var.
2. **Two supported modes during transition.** No-caching mode cannot be
   removed until every consumer migrates, so gates, docs, and support carry
   both matrices for at least one release cycle.
3. **PyTorch-version coupling enters the contract.** GPU-CR itself stays
   version-agnostic, but the *feature* now depends on `MemPool` semantics
   (≥ 2.5, with known behavioral drift since). The env var, for all its
   cost, worked on any PyTorch.
4. **Dump slack.** Segments carry cached-free bytes; a pathological
   allocate-free-allocate tenant pattern could inflate dumps beyond real
   state. Bounded by pool peak; measurable via the §8 metrics split
   (`storage_bytes` vs `device_bytes_freed`).
5. **Phase B adds a second allocation entry point** (pluggable path beside
   the hook), and divergence between them is a new bug class if not shared
   at the implementation level.

## Alternatives

1. **Status quo.** Keep the env var. Costs: process-wide allocation
   slowdown, 2.6× resident inflation at realistic scale, ~1500-call driver
   floors, region-count ceiling. These are the binding performance items in
   `future-improvements.md` (§3, §7).
2. **Sub-block (2 MB-chunk) release inside GPU-CR** — the "obvious"
   GPU-CR-only fix: allocate one `cuMemCreate` handle per 2 MB chunk so
   arbitrary chunk ranges can be unmapped. Rejected on three grounds:
   (a) the reuse hazard is untouched — the allocator will still place new
   blocks into parked bytes, and no preload hook can prevent a placement
   that involves no CUDA call; (b) LoRA tensors are mostly < 2 MB and
   unaligned, so under caching the fully-covered-chunk set — the only
   releasable set — is nearly empty: bytes dumped but almost no VRAM freed,
   defeating the feature's purpose (design doc §1.3); (c) it multiplies
   handle bookkeeping (~12,800 handles per 25 GB) and partial dump/restore
   layout complexity through `nv.cpp` and the dump format for a mode that
   still isn't correct.
3. **Replace the allocator wholesale** (`torch.cuda.change_current_allocator`
   → GPU-CR allocator). Without an internal caching layer this is the env
   var with extra steps (every tensor still a driver call); with one, GPU-CR
   grows a stream-aware caching allocator — months of work and a correctness
   surface PyTorch took years to stabilize. Strictly dominated by using
   PyTorch's own pools (Phase B gets the same tagging benefit at ~5% of the
   cost).
4. **Intercept PyTorch's VMM path under `expandable_segments:True`.** The
   dlsym/`cuGetProcAddress` interception layer in `ipc_hooks.cpp` could
   technically shim `cuMemCreate`/`cuMemMap` and gain chunk-level visibility
   of PyTorch's own VMM segments. Still rejected: the reuse hazard again
   (PyTorch believes those chunks are mapped and usable), plus fighting the
   allocator's own map/unmap lifecycle from below is exactly the fragility
   this project exists to avoid.
5. **Upstream PyTorch park/unpark API** (per-pool "release physical backing,
   keep VAs" + fault-on-touch). The cleanest end state — the allocator
   itself would guarantee no reuse — but it is a multi-quarter upstream
   effort with an unowned timeline. Worth pursuing in parallel; not a plan.

## Future Work

- Fold Phase B's tag registry into the KEP-0001 destination-path dumps
  (per-tenant dump files keyed by tag — one object per tenant, no shared
  scratch, which also closes `future-improvements.md` §1 for tagged
  tenants).
- Revisit `CUDA_LAUNCH_BLOCKING=1` (§7): with caching on, allocation-path
  interference is largely gone, which changes the repro conditions for the
  async-launch failure and may simplify root-causing it.
- Upstream-PyTorch park/unpark exploration (Alternative 5) as the eventual
  replacement for the preloader's release/remap entirely.
- AMD: evaluate MemPool-on-ROCm and the HIP equivalent of the hook path.

## Implementation History

- 2026-08-04 — 2 MB-floor inflation (1.47 GB → 3.8 GB) and ~1500-region
  driver-call floor measured on the llm-d multi-tenant rig; packing
  identified as the structural lever (`future-improvements.md` §3).
- 2026-08-05 — deep-dive confirming the reuse hazard makes a GPU-CR-only fix
  unsound; this proposal.
