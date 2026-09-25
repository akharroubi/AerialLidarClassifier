# Aerial LiDAR Classifier 1.1.0 — release remediation report

Date: 25 September 2026. Branch: `codex/release-hardening`.
Base: audited v1.1-dev commit `e1dda5d67bf291c7967a376f480a3d3154a2aff9`.

## Release decision

**The reproduced release-blocking code defects have been corrected in this
candidate. GO for an experimental public 1.1.0 release within the documented
validation scope; this is not a certification of every supported OS/GPU.**

The strongest validation is Windows 10, QGIS 3.44.10, Python 3.12 and NVIDIA
RTX 3090. Linux Python 3.10 installation was tested under WSL with QGIS
network/settings shims. macOS/MPS, physical low-memory GPUs, other GPU
generations, Python 3.13 and QGIS 4 were not validated in this remediation.
The metadata remains `experimental=True`; QGIS maximum version is now 3.99.

The maintainer confirmed SegFormer author permission and ownership of the
new LitePT trained weights. The report records that confirmation and retains
the existing weight licence; it does not claim independent examination of a
permission document or invent a broader grant.

Deliverables are in
`C:/Users/user/Desktop/QGIS_Plugin/release_candidate/dist/`:
`AerialLidarClassifier-1.1.0-rc1.zip` and its `.sha256` sidecar. They are newly
built artifacts. No existing QGIS publication, GitHub release, tag or live
QGIS installation was replaced. Historical 1.0.3 remains unpatched; these
fixes apply to the new candidate. The existing `v1.1` tag must not be treated
as proof that it contains this source.

The original independent audit remains unchanged at
[AUDIT_v1.1.0_ASTRA.md](C:/Users/user/Desktop/QGIS_Plugin/audit_20260925/AUDIT_v1.1.0_ASTRA.md).
This report supersedes its remediation status only for this candidate.

## 1. Disposition of all must-fix findings

| Audit finding | Correction and verification |
|---|---|
| R1: batch output overwrites another input | Validate every planned output against every input before loading a model. Resolve path aliases and existing hard links, and reject duplicate output names. Regression confirms the later input remains unchanged and no backend is created. |
| R2: scaled uint64 extra-byte corruption | Copy raw structured-array fields when streaming. Also fix the additional in-memory corruption path when adding a custom label dimension. Values above 2^53 and 2^63 retain exact stored integers. Both LAS and LAZ output paths are exercised. |
| R3: missing EVLRs/CRS | Explicitly write retained EVLRs after streamed point records. Ordinary EVLRs and a CRS stored only in an EVLR survive. COPC index records are removed because output is ordinary LAS/LAZ. |
| R4: truncated input reported successful | Require exactly the declared number of complete decoded records. Remove partial-EOF salvage. Truncation/decode errors stop processing; no shortened output is published. |
| R5: label field overwrites attributes | Protect standard LAS dimensions case-insensitively. Existing custom fields must be scalar, unscaled integers with sufficient range. Reject incompatible float, vector and scaled fields. |
| R6: custom fields lose raw classes | Custom fields receive model IDs. LitePT cars=3, trucks=4 and fences=6 remain distinct; only standard classification maps all three to ASPRS 1. Add a provenance VLR containing model, weight hash, encoding and class mapping. |
| R7: vertical EPSG ignored | Resolve GeoTIFF vertical CRS key 4096 when explicit vertical-unit key 4099 is absent. EPSG:6360 correctly yields US survey feet. Unknown vertical units require an explicit override. |
| R8: linear override accepts degrees | Detect geographic CRS before accepting a linear override. Metres/feet overrides cannot perform or bypass reprojection. |
| R9: absent headless Processing provider | Add idempotent `initProcessing()`. Test registry discovery and actual `qgis_process plugins enable` plus algorithm help using an extracted ZIP and scratch profile. |
| R10: insecure certificate retry | Remove TLS-downgrade commands and the insecure latch. Certificate failures remain failures, with CA/proxy repair guidance. Fault injection confirms no insecure retry occurs. |
| R11: automatic source builds | All install command builders use `--only-binary=:all:`. A real benign source-only fixture is refused and its build backend never executes. Replace the blanket NumPy<2 constraint so newer Python can resolve compatible wheels; this does not claim Python 3.13 was tested. |
| R12: broken extraction/partial Python | Validate internal link targets and feature-detect extraction filters. Handle an older distro filter that incorrectly resolves symlink targets from the archive root. Extract into staging, execute a runtime health check, then promote. A partial executable is not considered installed. Fresh Linux Python 3.10.16 extraction and installation succeed. |
| R13: LitePT falsely ready without spconv | Required spconv install/import failures cannot mark supported CUDA setup successful. LitePT's panel gate checks native dependencies and explains Repair/SegFormer alternatives. CUDA installation/verification no longer silently changes to CPU. Explicit CPU installs use PyTorch's CPU index on Windows/Linux. |
| R14: cached weights skip hash | Validate cached/migrated weights and rehash the same open file handle before `torch.load(..., weights_only=True)` in both backends. Tampered files are refused. Model promotion uses atomic replacement without first deleting good weights. |
| R15: SegFormer permission evidence | Maintainer confirmed author permission on 25 September 2026. Record its scope and provenance in THIRD_PARTY_NOTICES; add attribution in the adapted architecture file. The underlying permission record remains with the maintainer. Existing QGIS publication alone is not used as evidence of permission. |
| R16: weight licence inferred from DALES | Record the maintainer as creator of the LitePT weights. Keep the existing CC BY-NC 4.0 weight designation without asserting that the training dataset automatically imposes that licence. Retain upstream LitePT MIT notice and SegFormer attribution. |

## 2. Additional corrections and limits

Outputs are written to a temporary sibling file and replaced only after the
writer closes successfully. A cancelled or failed streaming write leaves an
existing good result intact. The temporary file has the intended LAS/LAZ
suffix, so compression selection is preserved.

Legacy formats requiring classification codes above 31 upgrade to a compatible
format: 0/1→6, 2/3→7, 4→9, 5→10. RGB survives; scan angles are converted to
the new unit and the exact old scan-angle rank is retained as an extra field.
Actual waveform packet payload relocation is not implemented. Files declaring
internal/external waveform payloads are explicitly refused, preventing outputs
with dangling packet references. Point-only data using waveform-capable formats
can still be processed.

LitePT's OOM retry loop now reaches the advertised 5,000-point floor. A synthetic
OOM regression forces 70,000→35,000→17,500→8,750→5,000 and returns a label for
every input point. This is a control-flow test, not a physical 5 GB GPU benchmark.
The small-card recipe and OOM recovery can change predictions because context
changes. A zero-metre tile buffer is no longer silently changed to 50 metres.

Model loading remains once per batch. Processing now defers loading until the
first prediction, avoiding a resident model while early input validation fails.
Streaming releases staging lists before model preprocessing. These are modest
resource improvements; the inference algorithm and normal recipe are retained.

Installers use cooperative cancellation; plugin unload no longer kills their
QThread with `terminate()`. Cancellation is checked before readiness is stamped.
Download workers survive parent-widget destruction, and normal dialog closing
does not block the UI waiting for the full download. Blocking QGIS network
requests themselves still cannot be interrupted halfway through a response.
Reinstallation cleanup runs in the worker. Replacing already loaded AI libraries
requires restarting QGIS and opening Repair before the classifier.

Transient `nvidia-smi` exceptions/timeouts are no longer cached permanently.
SOCKS proxy settings produce an explicit unsupported-proxy error instead of
being misrepresented as HTTP; NoProxy is respected. Corporate CA trust must be
configured correctly; the installer does not bypass verification.

## 3. Validation performed

**65 automated checks passed:** 42 existing/extended function checks, eight
new pure I/O/integrity regressions, and 15 QGIS release regressions. No model
smoke test was skipped in the final suite. Syntax checks and `git diff --check`
also passed. Additional diagnostic harnesses and real installers are separate
from that count.

| Validation | Observed result |
|---|---|
| Real SegFormer | CPU inference and CPU/CUDA agreement smoke checks pass. |
| Real LitePT | Actual CUDA model load and synthetic cloud classification pass using released SHA-pinned weights. Repeated successfully in the freshly installed torch 2.14 environment. |
| Pure release safety | Eight checks: batch/hard-link aliases, atomic failure, protected/incompatible fields, raw classes, scaled extras, legacy upgrades, corrupted weights. |
| QGIS release checks | Fifteen checks cover I/O and EVLRs, cancellation/truncation, installer policy, required spconv, tar links, partial Python, units, provider registration, batch preflight, atomic weight promotion, parent destruction, waveform refusal and cohort-link behavior. |
| Original audit fixtures rerun | PF1/6/8, scaled/vector extras, unit variants, legacy upgrades, corrupt tails and raw custom IDs show corrected behavior. |
| QGIS Processing | Single, tiled and streaming results expose OUTPUT_FILE and register layer loading after completion. Main-thread result loading creates a valid layer with a 3D renderer. |
| Modeler | Two dependent classification steps succeed; the second consumes the first OUTPUT_FILE. An initial harness crashed while configuring a detached QGIS model child; attaching the child before adding parameter sources fixes the harness and the rerun passes. No product change was needed for that harness misuse. |
| CLI | Extracted package enables as a Processing plugin and exposes `aeriallidar:classify_lidar` through qgis_process. |
| Fresh Windows CUDA install | Real QGIS APIs; Python 3.12.8, torch 2.14.0+cu126, torchvision 0.29.0+cu126, spconv 2.3.8, NumPy 2.5.2. CUDA kernel verification passes. About 478 seconds including downloads on this connection. |
| Fresh Linux CPU install | WSL Ubuntu, host Python 3.10.12, QGIS API shims. Final corrected run downloads/extracts standalone Python 3.10.16 and installs torch 2.14.0+cpu, torchvision 0.29.0+cpu, NumPy 2.2.6. Runtime reports CUDA build=None and CUDA unavailable. About 65 seconds on the successful run with network/cache conditions specific to this machine. |
| Host/venv interoperability | QGIS's already-loaded NumPy 2.4.4 remains active after adding the fresh environment; torch 2.14+cu126, spconv and scipy import, and NumPy↔tensor conversion succeeds. The venv is therefore storage isolation, not a separate inference process. |
| Real source-only dependency | Solver rejects the fixture because no usable wheel exists. `SOURCE_BUILD_EXECUTED False`. |
| UI | Panel and About rendered offscreen with Qt fonts and visually inspected. Link activation and persistent dismissal tested without opening a real browser. |

Evidence directory:
[release_evidence](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence).
Key logs:
[unit suite](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/unit_suite.log),
[QGIS regressions](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/qgis_checks.log),
[Windows installer](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/installer/windows_cuda.log),
[Linux corrected installer](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/installer/linux_cpu_retry.log),
[Modeler](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/integrity/modeler.log),
[CLI](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/cli_help.log).

## 4. Faster inference: what helps and what is still proposed

Speed, peak RAM, peak VRAM, desktop responsiveness and energy are separate
objectives. Lower peak memory does not necessarily make a run faster, and
lower instantaneous GPU use does not establish lower energy consumption.
No energy benchmark was performed.

Existing model reuse already avoids reloading the network per tile/file in a
batch. Existing tiling already limits the cloud passed to each model call.
Those features should be retained, not counted as new optimizations.

Measurements carried forward from the independent audit (not rerun as a new
large-data benchmark after remediation):

| Experiment | What it establishes |
|---|---|
| 3,281,702-point LitePT single-tile reference comparison | 30 differing labels, 99.99908584% agreement. This is implementation agreement, not ground-truth accuracy. |
| QGIS tiled vs streaming, same 150 m tiles/30 m buffer and recipe | 85.12 vs 85.52 seconds, excluding model load; raw predictions match exactly. Streaming is primarily a host-memory strategy here. |
| Buffered tile work | Four model inputs total 5,110,818 point visits, 55.7% above the raw source count. Halos repeat context work. |
| Smaller 35k-point/20 m recipe | 70.06 s versus 52.17 s for the normal 70k/30 m recipe in the corresponding single-file test. 14,670 labels change (0.4470%). Lower crop memory was slower on this example. |

Recommended next optimizations, in order:

1. Measure stage times and memory: decoding, partitioning, sidecar I/O, voxel/KD
   work, transfer, network inference, projection and writing. Avoid tuning GPU
   kernels when disk partitioning is dominant.
2. Implement indexed COPC reading beneath the existing tile/core/buffer contract.
3. Replace LAS/LAZ per-tile mask scanning with stable spatial bucketing, preserving
   each tile's point order. This targets partitioning overhead.
4. Assemble tiles into preallocated buffers rather than simultaneous lists and
   concatenated copies; use memory-mapped predictions and a coverage bitmap.
5. Benchmark a bounded preprocessing/prefetch queue with one GPU model worker.
   More workers can duplicate weights and exhaust VRAM.
6. Consider compilation, quantization or alternate attention implementations only
   as separate experiments with class-level agreement/accuracy measurements.
   Do not promise they are drop-in, numerically identical speedups.

Reducing buffers, crop radius, visits or model voxel resolution can save work
but changes context or geometry. Expose such choices explicitly with measured
quality tradeoffs; do not silently enable them as an equivalent fast mode.

## 5. COPC index use and very large LAS/LAZ

**Yes: use COPC's existing index. LAS/LAZ tiling already exists.** The next
improvement is a different reader underneath that existing tiler.

COPC's octree hierarchy identifies compressed chunks by location, file offset
and size. A reader can seek to overlapping chunks instead of scanning irrelevant
parts. laspy exposes bounded COPC queries, including resolution/level controls.
For full-resolution classification, do not request a reduced level of detail.
[COPC specification](https://copc.io/),
[laspy COPC API](https://laspy.readthedocs.io/en/latest/api/laspy.copc.html).

The audit's direct experiment used a real 27,163,160-point, 114.3 MB COPC file.
A central 100 m × 100 m window contained 308,274 points:

| Method | Two observed reads |
|---|---|
| Indexed full-resolution query | 0.150826 s; 0.096754 s |
| Sequential scan plus equivalent filter | 2.015270 s; 1.724328 s |

Ordered raw record bytes and the record multiset matched. These were warm-cache
spatial-read measurements. They do not establish a 13–18× whole-classification
gain: the current streaming pipeline partitions the file in one scan, rather
than rescanning the complete file for every tile. Full-file inference must
still read, classify and write every point.

A correct indexed implementation needs:

- Full-resolution buffered queries with bounds transformed from metres back to
  the source's horizontal coordinate units.
- Stable original point ordinals and original per-tile ordering, including
  duplicate coordinates. Geometry alone cannot identify duplicate records.
- Conservative boundary reads followed by the same exact buffer/core predicates.
- Each point committed once from its core, while halo predictions are discarded.
- Preservation of every non-label field and point order in the final writer.
- Equality tests for edge points, sparse/dense nodes, duplicate coordinates,
  feet/mixed units and complete point coverage, plus local/remote I/O benchmarks.

This reader has **not** been added to this release candidate. The existing
sequential COPC path is retained and hardened. Output remains ordinary LAZ;
regenerating a valid COPC index is a separate export operation.

Ordinary LAS/LAZ should keep sequential decoding because they do not supply a
COPC hierarchy. Their existing tiling works, but ordinary tiling reads the
whole input first. Enable streaming for files whose full records do not fit
RAM. Streaming still needs the largest buffered tile/model workspace and
disk staging; automatic tile sizing targets an average occupancy, not a hard
bound on dense hotspots.

Current global prediction plus inference-coverage storage is approximately
2 bytes per input point during inference: about 200 MB for 100 million points,
or 2 GB for one billion, before tiles, decoder and model workspace. Default
streaming chunks are five million points. Their raw storage alone is chunk
count × record size, with coordinate/mask/sidecar copies adding more. Scratch
XYZ/index staging is roughly 32 bytes per emitted membership before archive
overhead/compression, and buffer overlap increases memberships. These are
array-size estimates, not measured peak process RAM.

The next scalability release should expose host-RAM and scratch-space budgets,
chunk size and scratch location; check available disk before staging; count
buffered tile occupancy; and stop clearly if the selected recipe exceeds a
budget. A memory map/bitmap can remove the global in-RAM label/coverage burden.
Resume checkpoints need source/model/recipe hashes so stale partial work cannot
be mistaken for the current run.

## 6. Keeping the computer and GPU responsive

Available now: run one job at a time, use streaming for host-memory pressure,
avoid simultaneous point-cloud rendering/indexing, and disable automatic result
loading during long batches. Select SegFormer and disable GPU if the GPU must
remain free; CPU work and a different set of predictions remain. LitePT cannot
run on CPU. Clearing cached GPU memory does not unload active model tensors.

Recommended architecture: run inference in a supervised child process using
the managed interpreter. Keep Qt widgets, layers and project updates in QGIS;
exchange small progress/result messages and file paths. Set thread counts,
process priority and library environment variables in that child. This is safer
than changing process-wide PyTorch/OpenMP settings inside QGIS. PyTorch documents
that intra-op thread settings should be established before model work.
[PyTorch threading API](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html).

Candidate controls for that future worker are CPU thread count, host-RAM budget,
GPU allocation budget, bounded prefetch, pause/resume between crops and optional
pacing. Thread counts such as 2–4 are starting points to benchmark, not universal
defaults. A PyTorch allocator cap does not cover every CUDA allocation and does
not limit GPU utilization. Pacing can reduce continuous occupancy but lengthen
runtime; a paused model may still occupy VRAM. No such eco mode, hard utilization
cap or pause/resume feature is claimed in this candidate.

## 7. Cohort promotion implemented

The supplied course is
[LiDAR Point Clouds Processing in QGIS](https://maven.com/geomatics/qgis3d),
by Abderrazzaq Kharroubi. Its page describes live teaching on classification,
terrain products, COPC and 3D editing. The plugin uses that relevant offer without
embedding a price or cohort date that could become stale.

Placements:

| Placement | Purpose |
|---|---|
| Main panel, beneath run/progress controls | Persistent, contextual invitation: “Go further with LiDAR in QGIS.” CTA: “View syllabus & next cohort”. It can be dismissed and stays dismissed across sessions. |
| About dialog, after the plugin description | Full course title and concise learning outcomes, near the author/model context. |
| Plugin menu | “Live LiDAR course with the plugin author” remains available after panel dismissal. |

The invitation explicitly identifies optional paid training and says the plugin
remains free. It never opens a browser automatically or blocks classification.
Each clicked link carries only fixed UTM tags: source `aerial_lidar_classifier`,
medium `plugin`, campaign `qgis3d`, and content `panel`, `about` or `menu`.
No user identifier, filename, point-cloud information or background event is
sent by the plugin. These tags enable placement attribution if Maven's reporting
supports it; no conversion uplift or dashboard integration has been measured.

The main panel is the primary conversion placement because it is present while
users work with the tool. About/menu provide secondary discovery without repeated
popups. Review click-to-enrolment results on Maven before adding further prompts.

UI previews:
[panel](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/panel_preview.png),
[About](C:/Users/user/Desktop/QGIS_Plugin/release_candidate/release_evidence/about_preview.png).

## 8. Packaging and publication handoff

`python build_zip.py` produces a deterministic ZIP with one
`AerialLidarClassifier` root folder and a SHA-256 sidecar. Only runtime source,
assets, configuration, metadata and public documentation/licence notices are
included. Tests, local reports, evidence, environments and model weights are
excluded. Rebuilding identical source must yield an identical digest.

Use this new artifact for the experimental update, with release notes describing
the one-time installation-schema rebuild, LitePT's NVIDIA requirement, fixed
streaming integrity issues and the stated validation matrix. Do not overwrite
an older release's source tag to imply that it contained these fixes. Public
upload remains a separate action; this work prepared and tested the concrete
candidate locally.
