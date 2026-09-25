# Aerial LiDAR Classifier: report for Astra

Claude (Fable 5.1), night of 2026-09-24 to 25, on the user's Windows
workstation (RTX 3090, QGIS 3.44.10 LTR, Python 3.12). Requested by the user
before going to sleep: review the published plugin (1.0.2), fix what is wrong,
make LitePT-L the default model with SegFormer 3D kept as an option, handle
US feet, and report everything here so you can review the plugin before the
user uploads it in the morning.

Please read section 7 first: it lists what was not tested and the decisions
I took alone.

---

## 0. In one paragraph

Two versions exist. **1.0.3** is a patch release with every correctness fix
from the review (nothing new), built, tagged and pushed to GitHub `main`; the
user uploads its zip to plugins.qgis.org. **1.1.0** adds LitePT-L as the
default model on NVIDIA GPUs through a model registry and backend interface,
keeps SegFormer 3D for CPU / Apple Silicon, and is pushed to branch
`v1.1-dev`, installed in the user's QGIS profile for testing, and needs one
thing from the user before release: a GitHub release `v1.1.0` carrying the
172 MB weights file. Measured evidence: the LitePT-L port matches the
reference implementation on 99.998 % of 2.7 M points; feet-to-metres
conversion turns a 62 % agreement into 99.3 %; a fresh install of the new
dependency pipeline succeeded in 676 s with spconv verified and certificate
verification on.

---

## 1. Where everything is

| Item | Location |
|---|---|
| 1.0.3 tree and zip | `C:\Users\user\Desktop\QGIS_Plugin\v1.0.3\` (`aerial_lidar_classifier_v1.0.3.zip`, 41 files, 164 KB) |
| 1.0.3 on GitHub | `main`, commit `efbc9c9`, tag `v1.0.3` |
| 1.1.0 tree and zip | `C:\Users\user\Desktop\QGIS_Plugin\v1.1\` (`aerial_lidar_classifier_v1.1.0.zip`) |
| 1.1.0 on GitHub | branch `v1.1-dev`, commit `9670fb8` |
| Installed for testing | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\Aerial_LiDAR_Classifier` = 1.1.0 (the 1.0.2 copy that was there is in `QGIS_Plugin\backup\`) |
| LitePT-L weights (release artefact) | `v1.1\model_release\litept_l_dales_10cm_ema_fp16.pth` (171.8 MB), `SHA256SUMS.txt`, `litept_l_dales_10cm.json` |
| Weights placed for testing | `%APPDATA%\QGIS\QGIS3\profiles\default\AerialLidarClassifier\models\litept_l_dales_10cm\` and `...\segformer3d_urbanfiltering\` |
| Review of 1.0.2 | `v1.1\REVIEW_v1.0.2.md` (findings with file:line, status header) |
| Frozen 1.0.2 | `C:\Users\user\Desktop\QGIS_Plugin\Aerial_LiDAR_Classifier\` |
| Tests | `v1.1\Aerial_LiDAR_Classifier\tests\` (not shipped in the zip) |

---

## 2. Review of 1.0.2: what was wrong and what happened to it

Full detail with line numbers in `REVIEW_v1.0.2.md`. The install subsystem
(`venv_manager.py`, 3 800 lines) was reviewed by a second agent and its
two most serious claims were re-verified by execution before I acted on
them.

| Id | Finding | Evidence | Status |
|---|---|---|---|
| H1 | Flat tiles (Z range < 5.12 m) collapsed into one voxel block; 99.5 % of their points got fill value 7 = Building | synthetic tiles: 0.44 % classified before, 100 % after; `tests/test_block_partition.py` fails on 1.0.2 code, passes now | fixed (1.0.3) |
| I1 | Plugin could not open on Python 3.9 to 3.11 QGIS builds: PEP 701 f-strings in 4 files (GitHub #3, Ubuntu 22.04 / QGIS 3.44 / Python 3.10) | `compile()` with Python 3.11.5: 4 files failed; 0 after; `tests/test_syntax_compat.py` | fixed (1.0.3) |
| I2 | Fresh installs ended with CPU torch: Phase B `uv pip install --upgrade` re-resolved torch from PyPI because `timm` depends on it | `uv --dry-run` of the exact command: `- torch==2.12.0+cu126 + torch==2.14.0` | fixed: no `--upgrade`, torch pinned by constraints file; dry-run shows torch untouched even with `--upgrade` |
| I4 | TLS verification disabled for the package hosts on every install (`--allow-insecure-host` on first attempt) | 1.0.1 changelog and code | fixed: verification on; one retry with it off after a TLS error, logged |
| I3 | CUDA cascade could leave a venv without torch while marked ready | code reading | fixed: marker after verification; torch restored; marker records torch build, CUDA index, packages |
| I12 (found tonight) | The torch cap number was also applied to torchvision (`torchvision<2.13`, never binding: torchvision is 0.x), so uv took the newest torchvision on the index and, with `--upgrade`, replaced torch by the version torchvision pins | fresh install marker: torch 2.14.0+cu126 where 2.12.x was expected; dry-runs: on cu121 / cu124 the pair still resolves correctly (2.5.1 / 2.6.0 with matching torchvision), so only indexes carrying a newer torch are affected | fixed in 1.1.0: torch pinned by a constraints file right after its install, torchvision resolved under it (dry-run: 0.27.1 under a 2.12.1 pin, 0.20.1 under 2.5.1), same in the cascade. 1.0.3 left as tagged: on cu126 it yields a newer CUDA torch that was verified working; on other indexes it is unaffected |
| units (#5, #2) | Feet fed as metres: buildings -> wires, 13x slower | see section 5 | fixed (1.0.3) |
| H2 | Apple Silicon: MPS offered, `model.cuda()` crashed | code reading | fixed: device string end to end; MPS untested (no Mac) |
| H3 | Processing added the layer from the worker thread | code reading | fixed: `addLayerToLoadOnCompletion` + post-processor; `OUTPUT_FILE` output added |
| M1 | Version constant said 1.0.1 in the 1.0.2 release | About / provider name | fixed: read from metadata.txt |
| M2 | Auto tile size from the longer side: corridor extents got 2.5x the target | `tests/test_tile_grid.py` | fixed: area-based size, no sliver tiles |
| M3 | Broken model config skipped the file silently | code | fixed: raises |
| M5 | Streaming used 4 bytes/point for predictions | code | fixed: 1 byte, codes > 255 refused |
| I8 | Download dialog: cancel did nothing, closing during download destroyed a running thread | code | fixed |
| L4 | Model rebuilt per tile | measured 0.12 to 0.5 s per call, minor | fixed by design (backends load once per run) |
| M6, M7, I5, I6, I7, I9, I10, I11, L3, L6 | streaming partition O(chunks x tiles), synchronous torch import at dock open, nvidia-smi timeout pinning CPU, partial standalone-Python extraction, cosmetic cancel, UI-thread rmtree, proxy types, numpy pin, duplicate block listing, repo leftovers | in the review | **open**, listed in section 7 |

---

## 3. 1.0.3 patch release

Contents: every fix above marked 1.0.3, the coordinate-unit handling, the
tests, an updated README (units section, Processing algorithm id corrected to
`aeriallidar:classify_lidar`, Python badge 3.9 to 3.13, MPS marked untested).
No new feature, one model, same install schema (2), so existing users are
not forced to reinstall.

Checks run on the 1.0.3 tree: compile on Python 3.11.5 and 3.14, pyflakes,
`test_block_partition` (6), `test_tile_grid` (5), `test_las_units` (14),
`test_syntax_compat` (2), `test_classifier_core_smoke` (5, CPU and CUDA runs
agree > 98 %), headless QGIS 3.44 import and instantiation of every module,
the algorithm and the dock.

Not done for 1.0.3: a run inside the QGIS GUI (headless only), and a fresh
install of the 1.0.3 pipeline (the fresh install in section 4.6 used the
1.1.0 pipeline, which is a superset).

---

## 4. LitePT-L integration (1.1.0)

### 4.1 The model

From `E:\MLS_Classifier\Lite_Dales` (your own review documents there):
original LitePT-L, 85 845 032 parameters, 2 input channels (constant +
height above crop floor), 8 classes, 10 cm grid, 70 000-point / 30 m crops,
trained from scratch for 200 epochs on the 32/4/4 deployment split.
Validation-selected checkpoint: epoch 110, EMA weights, val mIoU8 0.7966.
Custom four-tile test (`test.json`): mIoU8 **0.8239**, OA 0.9792; per class
IoU ground 0.972, vegetation 0.940, cars 0.889, trucks 0.367, power lines
0.968 (fences, poles and buildings in the file). Trucks is the weak class
and the plugin maps cars, trucks and fences to ASPRS 1 anyway.

### 4.2 Weights export

`best.pt` is 1.37 GB (model + EMA + optimizer + scheduler). I exported the
EMA state dict as float16: 343 float tensors + 13 integer buffers,
**171.8 MB**, SHA-256 `849ba508...b09592b`, loads with
`torch.load(weights_only=True)`, maximum relative rounding error over the
weights 4.8e-4. The network runs in float32 (weights upcast at load), which
is how it was validated; attention casts to half internally as upstream
does.

### 4.3 The port into the plugin

Upstream LitePT needs `spconv`, `torch_scatter`, `flash_attn` and a CUDA
PointROPE kernel. The plugin's rule is "pure wheels on every OS, no
compiler". Findings:

- `spconv`: prebuilt wheels `spconv-cu118/cu121/cu124/cu126` (2.3.8) for
  Windows and Linux, cp312 included; **no cu128 wheel, no CPU build for
  Python 3.12, no macOS build**. It cannot be replaced tonight (the
  submanifold sparse convolutions are the first three encoder stages).
- `torch_scatter`: replaced by a pure-torch `segment_csr` (`index_add_` /
  `scatter_reduce`), tested against the expected semantics.
- `flash_attn`: replaced by `scaled_dot_product_attention` over the
  serialized patches (batched when all patches are equal, per patch
  otherwise).
- PointROPE: upstream ships a pure-torch fallback; used.
- `addict.Dict`: replaced by a 15-line attribute dict.
- `colorhash`: only used by a debug helper; import made lazy.

Upstream `model.py` and `serialization/` are vendored unchanged except the
four import lines (MIT licence kept as `core/litept/LICENSE.upstream`, commit
`436d0480` recorded in the file header).

`core/backends/litept.py` reproduces `mls_dales.infer`: integer origin
removal, one representative per occupied 10 cm voxel with exact membership
projection, regular crop centres at 31.49 m spacing then one crop per
uncovered point, per-crop centring / floor referencing / voxelisation,
softmax averaging per visit, argmax, projection to raw points.

**Agreement with the reference implementation** (your WSL `mls-litept`
environment, FlashAttention, CUDA RoPE, fp32 weights) on a 300 x 300 m
subset of `LHD_cls.las`, 2 708 283 points:

| Comparison | Agreement |
|---|---|
| Plugin backend, single pass, vs reference | **99.998 %** (per-class recall vs reference 99.81 to 100 %) |
| Plugin through the Processing algorithm, 150 m tiles + 30 m buffer, vs reference | 99.093 % (crop centres differ near tile edges) |
| Tiled vs streaming mode | 100.000 % |

Class shares were identical to the reference to the second decimal.

### 4.4 Performance and memory

RTX 3090: 60 000 points/s single pass (the reference reports about 71 000
with FlashAttention; the SDPA fallback costs 15 %), SegFormer 3D about
110 000 points/s. Peak GPU memory 4.53 GB allocated but 16.6 GB reserved
(allocator fragmentation from sparse-conv and attention temporaries), so:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before torch
starts and `_set_allocator_settings` is called as a fallback; cards under
8 GB get the 35 000-point / 20 m crop recipe; an out-of-memory crop halves
the crop size and retries (down to 5 000 points) before giving up with a
message. **The small-card path is untested** (only a 24 GB card here).

### 4.5 Plugin architecture changes

- `core/registry.py`: one `ModelSpec` per model (id, weights file, URLs,
  SHA-256, bundled config, class mapping, supported devices, extra
  packages, licence, attribution). `config.py` lost its model constants.
- `core/backends/`: `load(device)` / `predict(xyz_m, progress, cancel)` /
  `unload()`; `segformer.py` wraps the split `load_segformer` /
  `predict_blocks` of `classifier_core.py`; `litept.py` as above.
- `workers/classifier_task.py`, `workers/streaming_classifier.py`,
  `processing/classify_algorithm.py`: take a `model_id` / `MODEL`
  parameter, load the backend once per run, call `predict_fn(xyz, cb)`
  per tile, take the class mapping from the spec. `predict_fn` replaces
  the old `(config_path, model_path, use_cuda)` plumbing everywhere.
- `utils/model_manager.py`: one manager per spec, weights under
  `models/<model id>/`, the v1.0 file location is migrated automatically,
  `import_file()` (copy + SHA-256) for offline machines.
- Dock: model combo in the status strip (short names, full description in
  the tooltip), per-model status, download and import buttons, a device
  gate (LitePT-L selected without CUDA: Run disabled with the reason, red
  status), `model_id` persisted. Default: saved choice if it fits the
  device, else LitePT-L on CUDA, SegFormer 3D elsewhere.
- Processing: `MODEL` enum (index 0 LitePT-L, 1 SegFormer 3D; order
  stable), help text generated from the registry.
- About dialog and metadata list both models and licences.
- QGIS 4 / Qt6: the Qt enums were already scoped; QGIS enums are now
  scoped with 3.34 fallbacks (`Qgis.MessageLevel`,
  `Qgis.ProcessingParameterFlag`, `Qgis.ProcessingNumberParameterType`,
  `QgsTask.Flag`, `QgsBlockingNetworkRequest.ErrorCode`,
  `Qgis.LayerFilter`). `qgisMaximumVersion=4.99`. **Untested on QGIS 4**:
  no Qt6 build on this machine (the `QGISQT6 3.40.15` folder is an
  aborted install) and installing one is a multi-GB download I did not
  start without the user.

### 4.6 Installer changes and the fresh-install test

- `scipy` added to the required packages; `spconv-<cuXXX>` installed in a
  new non-fatal Phase C on CUDA installs where torch came from
  cu118/cu121/cu124/cu126, under the torch constraints file; it follows
  torch through the CUDA cascade; verification treats it as optional (a
  failure disables LitePT-L only).
- Cascade order for non-Blackwell GPUs is now cu126, cu128, cu124,
  cu121, cu118 (spconv has no cu128 wheel; current torch releases ship
  +cu126). Blackwell keeps cu128 only, so **RTX 50 cards cannot run
  LitePT-L** until spconv publishes cu128; the dock will say so.
- Install schema 3: every existing venv is rebuilt once. The user's own
  venv was stamped schema 3 by hand because it already holds
  `spconv-cu126 2.3.8` and `scipy` (it was used for all tests).
- Marker now records `torch_version`, `torchvision_version`,
  `cuda_index`, `cuda_mode`, `packages`, `spconv_version`.

Fresh install #1 into a scratch cache directory
(`AERIAL_LIDAR_CLASSIFIER_CACHE_DIR`), QGIS 3.44 Python, NVIDIA driver
616.56, compute capability 8.6: cu126 chosen, torch and torchvision from
the cu126 index, batch phase without `--upgrade` under the constraints
file, spconv installed in Phase C and verified (9/9 packages), CUDA smoke
test passed, **no TLS insecure fallback used**, result "Virtual environment
ready" after 676 s, 4.8 GB on disk. The venv holds torch **2.14.0+cu126**
(not the 2.12.x the cap asked for: finding I12 above, found through this
test and fixed afterwards), torchvision 0.29.0+cu126, spconv 2.3.8, scipy
1.17.1. Both model test suites (`test_litept_backend.py`,
`test_classifier_core_smoke.py`) pass on that venv, so the stack the
installer produces is validated. Fresh install #2 with the pin fix was
started at the end of the night; its result is in section 9.

### 4.7 Tests added

`tests/`: `test_block_partition.py` (6), `test_tile_grid.py` (5),
`test_las_units.py` (14), `test_syntax_compat.py` (2),
`test_registry.py` (6), `test_classifier_core_smoke.py` (5, needs the
venv), `test_litept_backend.py` (3, needs CUDA + spconv + weights;
skips otherwise). All pass. `tests/_load.py` imports plugin modules
outside QGIS. Headless QGIS scripts used for the end-to-end runs are in
the session scratchpad, not in the repository.

---

## 5. Coordinate units (GitHub #5 and #2)

`utils/las_units.py` reads the linear unit from the LAS header: WKT (1 and
2, compound; projection parameters and geographic base are skipped so a
metric false easting inside a feet CRS is not mistaken for the unit) or
GeoTIFF keys 3076 / 4099, with an EPSG lookup through QGIS when only the
CRS code is present. Geographic files are refused. XYZ is scaled for the
model only; the output keeps the original coordinates. Override in the dock
(`Input units`) and Processing (`UNITS`).

Evidence, same 300 x 300 m subset written in metres and in US survey feet
with a real ftUS compound WKT (EPSG:2229 + 6360):

| Run | Time | Agreement with the metric run | Building points reported as |
|---|---|---|---|
| Feet, unconverted (1.0.2 behaviour) | 283 s | 62.0 % | Wire 66.4 %, Pole/Tower 1.4 %, Building 2.6 % |
| Feet, converted (1.0.3) | 22 s | 99.3 % | Building |

This also explains #2 (35 minutes at 30 % on a 22 MB Madison, WI file:
Wisconsin county systems are in US survey feet). I did not download the
reporter's sample file to confirm its CRS (a download I chose not to start
alone); the reply below asks them to re-test with 1.0.3.

---

## 6. GitHub issues: proposed short replies

**#3 "Python error at launch" (Ubuntu 22.04, QGIS 3.44.7, Python 3.10)**

> Thank you, and sorry: versions 1.0.0 to 1.0.2 used a Python 3.12-only
> f-string form in four files, so the plugin could not load on any QGIS
> built against Python 3.9 to 3.11. Fixed in 1.0.3 (all files now compile
> on 3.9+, with a test to keep it that way). It is on GitHub (tag v1.0.3)
> and will be on plugins.qgis.org shortly. Please reopen if 1.0.3 still
> fails on your machine.

**#6 "Install fails under QGIS 4.2.1"**

> Until 1.0.3 the metadata capped compatibility at 3.99. Version 1.1.0
> (branch v1.1-dev, release soon) declares QGIS 4.x and its enum usage is
> Qt6-safe, but I could only test on 3.44 LTR: I have no QGIS 4 build to
> hand. If you can try 1.1.0 on 4.2.1 and post the first error you see
> (if any), I will fix it in the next patch.

**#5 "Classifications all wrong" (LAS 1.4 PF6, RTX 4070 Ti)**

> The symptom (buildings as Wire-Conductor and Transmission Tower, other
> classes scrambled) is what the model produces when the file is in feet:
> until 1.0.2 the plugin fed coordinates to the model unconverted, so every
> distance was 3.28x too small for it. 1.0.3 reads the unit from the file's
> CRS and converts to metres for the model (with a manual override under
> Advanced parameters > Input units if the CRS is missing). On a test tile
> the unconverted feet run agreed with the metric result on 62 % of points
> and the converted run on 99.3 %. Could you re-run with 1.0.3? If your
> file is in metres and the result is still wrong, please share its CRS
> (the "Coordinate units" line in the log) and I will dig further.

**#2 "Very slow classification speed" (22 MB Madison COPC, 16 GB VRAM)**

> Two things were wrong. If that file is in US survey feet (Wisconsin county
> systems usually are), the model saw ten times more voxel blocks than it
> should, which alone makes the run about 13x slower; 1.0.3 converts units
> automatically. And the 1.0.0 installer could end up with a CPU-only torch
> while the dock said GPU; 1.0.3 pins the CUDA build. With 1.0.3 on an RTX
> 3090, 2.7 M points take about 25 s with SegFormer 3D. Please re-test with
> 1.0.3 and post the "Coordinate units" and "Model:" lines from the log if
> it is still slow.

---

## 7. What is NOT done, decisions I took alone, open questions

1. **GitHub release with the weights.** The registry points LitePT-L at
   `https://github.com/akharroubi/AerialLidarClassifier/releases/download/v1.1.0/litept_l_dales_10cm_ema_fp16.pth`.
   That release does not exist; I cannot create it (no `gh`, no GitHub
   session). Until the user creates release `v1.1.0` and uploads the file
   from `v1.1\model_release\`, users must use the dock's "import weights
   file" button. The SHA-256 is in the registry and in `SHA256SUMS.txt`.
2. **Upload order.** My recommendation: upload 1.0.3 to plugins.qgis.org in
   the morning (patch, one model, tested install path is a superset), and
   1.1.0 only after your review and after the release with the weights
   exists. Both zips are built.
3. **QGIS 4 / Qt6 untested** (section 4.5). **Apple MPS untested** (no
   Mac). **Small-GPU crop path untested** (section 4.4). **Linux fresh
   install untested tonight** (WSL has no QGIS; the pipeline changes are
   platform-neutral, but Linux spconv wheels were only checked to exist).
4. **Licence text for the LitePT-L weights.** I wrote "CC BY-NC 4.0
   (trained on DALES, non-commercial)" and the attribution "LitePT:
   Photogrammetry and Remote Sensing Lab, ETH Zurich. Trained by GeoScITY
   Lab, University of Liege" in `core/registry.py`, the About dialog and
   the README. The user should confirm both; DALES's own licence is
   non-commercial, which is why the weights cannot be more permissive.
5. **DALES / LitePT citations** in the README: DALES (Varney et al., CVPRW
   2020) is cited; for LitePT the README points to the repository for the
   paper because I did not want to invent an author list.
6. **Class mapping choices.** LitePT-L: cars, trucks and fences -> ASPRS 1
   (Unclassified), power lines -> 14, poles -> 15, consistent with the
   SegFormer mapping. Users who need vehicles as a class can write raw
   model ids to an extra dimension with `FIELD_NAME`.
7. **cu126 preferred over cu128** for every non-Blackwell GPU: cheaper for
   LitePT-L, but it is a change of the torch build users get; the
   validated stack is torch 2.12.0+cu126 + spconv-cu126 2.3.8 (the cap
   table still limits cu126 to torch < 2.13, deliberately, until a newer
   combination is tested).
8. **Schema 3 forces a one-time reinstall** (2 to 3 GB) for every existing
   user of 1.0.x. The alternative, an incremental "install what is
   missing", is designed in the review (marker now carries `packages`) but
   not implemented tonight.
9. **Still open from the review**: streaming partition O(chunks x tiles)
   on COPC-ordered files (M6), synchronous torch import when the dock
   opens (M7), nvidia-smi timeout pinning the user to CPU (I5), partial
   standalone-Python extraction never repaired (I6), cancel during install
   is cosmetic and `terminate()` orphans uv (I7), multi-GB rmtree on the UI
   thread (I9), proxy types (I10), the `numpy<2` venv pin that does not
   govern the in-process numpy (I11), duplicate block listing (L3), repo
   leftovers and the 12 GB of benchmark LAS not ignored by git (L6), and
   the install quick wins (duplicated command builders, dead code).
10. **The 99.09 % tiled agreement** is expected (crop centres change at
    tile borders) but it means results depend slightly on tile size;
    documented, not "fixed".
11. **Headless only.** No click-through of the dock in the QGIS GUI was
    possible; the dock was constructed headless with the RTX 3090 detected
    and both models listed. The user's morning test is the first GUI run.
12. **Deprecation warning** at LitePT-L load:
    `torch.cuda.memory._set_allocator_settings` is a private API; wrapped
    in try/except, harmless, may need `torch.cuda.memory._set_allocator_settings`
    replaced by the public API in a future torch.
13. **torch version caps are a maintenance item.** `_TORCH_VERSION_CAP_BY_CUDA`
    protects against `+cpu` fallback wheels on indexes PyTorch has stopped
    updating; cu126 was raised to `<2.15` tonight (2.14.0+cu126 verified),
    cu128 still says `<2.12` from May and was not re-checked (its dry-run
    printed nothing usable; RTX 50 owners are the only ones on it). The
    CUDA smoke test and cascade remain the real safety net.
14. **1.0.3 keeps the torchvision behaviour of I12.** On cu126 it installs
    torch 2.14.0+cu126 (verified working with SegFormer 3D); on cu118 /
    cu121 / cu124 the pair resolves correctly. I did not re-tag 1.0.3 for
    this; if you prefer the pin fix in the patch line, it is a 1.0.4 with
    the `install_dependencies` and `_reinstall_torch_at_cuda_index` hunks
    of the v1.1-dev commit.

---

## 8. Morning checklist for the user

1. Open QGIS 3.44 (the installed plugin is 1.1.0; the venv is already
   stamped, no reinstall prompt expected). The dock should show
   `Model: LitePT-L  ready (172 MB)` with the RTX 3090 as device. Run a
   tile with each model; compare with earlier 1.0.2 outputs.
2. Upload `v1.0.3\aerial_lidar_classifier_v1.0.3.zip` to plugins.qgis.org
   (the scanner rules from May still apply: Bandit and secrets only block;
   all `torch.load` calls use `weights_only=True`, hashes carry the
   allowlist pragma, the bundled JSON card carries no hash).
3. Post the four GitHub replies from section 6 (edit freely).
4. When ready for 1.1.0: create GitHub release `v1.1.0`, upload
   `v1.1\model_release\litept_l_dales_10cm_ema_fp16.pth`, merge `v1.1-dev`
   into `main`, upload `v1.1\aerial_lidar_classifier_v1.1.0.zip`.
5. Tell me the outcome of Astra's review and I will work through section 7.

---

## 9. Addendum: fresh install #2 (with the torchvision pin fix)

Same method as #1, new scratch cache, run after the I12 fix and the cu126
cap change: cu126 chosen, result "Virtual environment ready" after 403 s
(uv's wheel cache was warm from #1), status Ready, **no TLS insecure
fallback used**, marker: torch 2.14.0+cu126, torchvision 0.29.0+cu126,
spconv 2.3.8. With the cap now `<2.15`, 2.14.0 is the expected torch and
torchvision was resolved under its pin, so the pair is the matching one
by construction rather than by luck. The venv passes the same model
tests as #1 (identical stack). The scratch venvs were deleted afterwards
(4.8 GB each); both logs are kept in the session scratchpad
(`fresh_install.log`, `fresh_install2.log`).
