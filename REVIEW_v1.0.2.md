# Aerial LiDAR Classifier v1.0.2: code review

**Status on 2026-09-25 (end of day).** Fixed in 1.0.3 (released, tag
`v1.0.3` on GitHub main) and carried into 1.1.0: H1, H2, H3, M1, M2, M3,
M4, M5, L1 (raises instead of promising a fallback), L2, L4 (backends load
once per run), L5 (registry), I1, I2, I3, I4, I8 (download dialog: cancel
disabled during the blocking download, thread waited on close). Still
open, tracked in `REPORT_FOR_ASTRA.md` section 7: M6, M7, L3, L6, I5, I6,
I7, I9, I10, I11, the install quick wins.

Date: 2026-09-25. Reviewed copy: `Aerial_LiDAR_Classifier/` (frozen v1.0.2,
identical to the GitHub main branch up to CRLF). Working copy for the next
version: `v1.1/Aerial_LiDAR_Classifier/`.

Method: every file of the classification pipeline, the dock, the Processing
algorithm and the streaming path was read in full. Two claims were checked
numerically in the plugin venv (`~/.qgis_aerial_lidar_classifier/venv_py3.12`):
the block partitioning on flat tiles (section 2, H1) and the cost of
rebuilding the model per tile (L4). The install subsystem
(`utils/venv_manager.py` and friends, 144 KB) was reviewed separately;
its findings are in section 5.

Line numbers refer to the v1.0.2 files.

---

## 1. Architecture as it stands

```
plugin.py            toolbar/menu, dependency gate (venv_manager.get_venv_status)
  └─ gui/main_panel.py            dock: file list, output, advanced, log, run
       └─ workers/classifier_task.py   QgsTask; per file: read -> infer -> map -> write
            ├─ core/classifier_core.py     filterPoints(): build model + load weights
            │                               + voxelise + infer + argmax, all in one call
            └─ workers/streaming_classifier.py  4-pass disk-backed variant
processing/classify_algorithm.py   re-implements the per-file logic for the Toolbox
utils/model_manager.py             one model: path, download, SHA-256
config.py                          one model: filename, URLs, hash, class mapping
```

Torch and laspy are imported **in-process** into QGIS's Python: the venv's
`site-packages` is appended to `sys.path` (`ensure_venv_packages_available`).
The venv isolates the *install*, not the *runtime*. Consequences that matter
for v1.1: the venv's Python minor version and numpy ABI must match QGIS's,
torch can never be unloaded, a CUDA fault takes QGIS down with it, and any
model whose dependencies pin a different torch/numpy cannot coexist.

---

## 2. Bugs, ranked

### H1. Flat tiles are classified almost entirely as Building (verified)

**Status: fixed in `v1.1/` on 2026-09-25** (`core/classifier_core.py`,
grid clamped to at least one block per axis, fill value 0 instead of 7;
regression test `tests/test_block_partition.py`, 6 cases, all pass on the
fixed code and the flat cases fail on the v1.0.2 code).

`core/classifier_core.py:41-42`:

```python
steps = np.floor((p_max - p_min - bs) / stride).astype(int) + 1
dims = steps + 1
```

When a dimension's range is smaller than 10 % of the block size, `steps`
goes negative and `dims` reaches 0. The plugin always calls
`filterPoints(..., if_bottom_only=False)`, so blocks are 3D and the Z block
is 256 x 0.2 m = 51.2 m. Any point cloud (whole file, or one tile + buffer in
tiled/streaming mode) with a **Z range under 5.12 m** hits this:
`np.clip(..., 0, -1)` returns -1 for every point, all points land in a
single block, and the voxel filter `nb_sel` (line 198) then keeps only the
33.6 x 33.6 m corner nearest the minimum. Every other point keeps the fill
value `num_classes - 1 = 7` (line 215), which the class mapping turns into
**Building (ASPRS 6)**.

Measured on a synthetic 500 x 500 m tile, 300 k points, uniform Z:

| Z range | blocks | points classified | points left at fill value 7 (Building) |
|---|---|---|---|
| 3 m | 1 | 0.44 % | 99.56 % |
| 5 m | 1 | 0.45 % | 99.55 % |
| 6 m | 289 | 100 % | 0 % |
| 40 m | 289 | 100 % | 0 % |

numpy also emits `divide by zero encountered in floor_divide` (line 92),
which is the only visible symptom, and it goes to stderr, not the QGIS log.

Who hits it: flat farmland, water, airports, polders, any treeless tile.
In streaming mode over a large rural file, individual tiles can be flat even
when the file as a whole is not, so the output would show rectangular
"Building" patches.

Fix (two parts, both small):
1. Clamp the grid to at least one block per axis:
   `steps = np.maximum(steps, 0)` before `dims = steps + 1`. With `dims_z = 1`
   the single Z block spans `[p_min_z, p_min_z + 51.2)`, which covers a flat
   tile completely.
2. Change the fill value for points that never enter any block from `7`
   (Building) to `0`. Model id 0 is not in the class mapping, so the
   existing "unmapped ids -> ASPRS 0 + warning" path (`classifier_task.py:452-462`)
   makes any leftover visible instead of silently inventing buildings.
3. Add a regression test: synthetic flat tile must classify 100 % of points.

### H2. Apple Silicon: "Use GPU" is offered, then crashes at run time

`utils/helpers.py:124-134` reports MPS as an available GPU. The dock then
enables and checks "Use GPU (Apple Silicon (MPS))" (`gui/main_panel.py:804-813`)
and passes `use_cuda=True` to the task. `core/classifier_core.py:155-157`
only knows two devices and calls `model.cuda()`, which raises
`AssertionError: Torch not compiled with CUDA enabled` on every Mac.
`metadata.txt` advertises MPS support.

Fix: replace the `use_cuda: bool` that is threaded through the task, the
streaming path and the Processing algorithm with a `device: str`
(`"cuda"`, `"mps"`, `"cpu"`) and use `model.to(device)` /
`x.to(device)`. Conv3d on MPS was missing in older torch releases; test on
an M-series machine, and if it fails, fall back to CPU **with a visible
message**, or drop the MPS claim from metadata.

### H3. Processing algorithm modifies the project from a worker thread

`processing/classify_algorithm.py:553-569` creates a `QgsPointCloudLayer`,
calls `QgsProject.instance().addMapLayer()` and sets the 3D renderer inside
`processAlgorithm()`. The Toolbox runs `processAlgorithm()` in a background
thread (the algorithm does not set `FlagNoThreading`), and project/layer
mutation from that thread is documented as unsafe; it works often and
crashes occasionally, typically on the 3D renderer or at project save.

Fix: register the result with
`context.addLayerToLoadOnCompletion(path, QgsProcessingContext.LayerDetails(name, context.project(), OUTPUT_FILE, QgsProcessingUtils.LayerHint.PointCloud))`
and attach the 3D renderer in `postProcessAlgorithm()` (main thread). While
there, return the output **file** as a proper output parameter, not only the
folder (`return {self.OUTPUT_FOLDER: ...}` at lines 416 and 539): today the
Graphical Modeler cannot chain the classified file into a next step.

### M1. Version constant out of sync with metadata

`config.py:15` says `1.0.1`, `metadata.txt:6` says `1.0.2`. The About dialog
(`dialogs/about_dialog.py:48`) and the Processing provider name
(`processing/provider.py:35`, shown as "Aerial LiDAR Classifier (v1.0.1)" in
the Toolbox) both report the wrong version for the shipped release. The venv
install marker records it too, but only for diagnostics
(`venv_manager.py:3364-3373`), so no user was forced to reinstall.

Already set to `1.1.0` in the working copy. The durable fix is to read the
version from `metadata.txt` at import time so there is one source of truth.

### M2. Auto tile size ignores the extent's aspect ratio

`workers/classifier_task.py:119-121` and `workers/streaming_classifier.py:243-245`:
`tile_size = max(width, height) / ceil(sqrt(n_tiles_needed))`. For a
corridor-shaped file (10 km x 0.5 km, 100 M points): 10 tiles wanted,
`side = 4`, tile = 2500 m, grid = 4 x 1, so each tile carries ~25 M points
instead of ~10 M. In streaming mode this is exactly the memory bound the
mode exists to guarantee. Fix: `tile_size = sqrt(width * height / n_tiles_needed)`.

### M3. A model-config error silently skips the file

`core/classifier_core.py:123-129` prints to stdout and returns `None` when
`model_config.json` cannot be loaded. Callers treat `None` as "no
predictions" (`classifier_task.py:440-442`, `streaming_classifier.py:553-557`),
log a warning and move on; the dock then reports "Complete - N-1 file(s)
processed". Raise instead; there is no valid outcome without the config.

### M4. Misleading log in the tiled merge

`workers/classifier_task.py:223-231` logs "Filling N unassigned point(s) via
nearest tile prediction" and then sets them to 0. With the half-open core
and the epsilon nudge this branch should never fire; if it does, something
upstream is wrong. Say what actually happens ("set to ASPRS 0"), and count
it in the same "unmapped" warning as H1.

### M5. Streaming holds 4 bytes per point where 1 is enough

`workers/streaming_classifier.py:355` allocates `predictions` as `int32`
for the whole file. The class mapping only ever produces ASPRS codes
(0-255), and the custom extra-byte field receives the same values. `uint8`
cuts the largest fixed allocation of the streaming path by 4x (a 2 G-point
file: 8 GB -> 2 GB). Keep `int32` only if a future mapping can emit codes
above 255.

### M6. Streaming partition is O(chunks x tiles)

`workers/streaming_classifier.py:465-477` tests every chunk against every
tile. The bbox pre-check (line 466) only helps when chunks are spatially
compact (flightline order). COPC and other octree-ordered files spread each
chunk over the whole extent, so every chunk masks every tile: for a 5 G-point
file (500 tiles, 1000 chunks) that is 500 000 mask passes over 5 M points.
Fix: compute each point's tile column/row with
`floor((x - xmin) / tile_size)`, then add the point to the up-to-8 neighbour
tiles whose buffer it falls in. O(points), independent of tile count.

### M7. First dock open freezes QGIS while torch loads

`gui/main_panel.py:129` calls `_check_gpu()` in the constructor, which
imports torch and initialises CUDA synchronously (2-10 s on a cold start,
longer on slow disks or corporate AV). The "Device: detecting..." label
exists but is never seen. Run the probe in a QgsTask or QThread and update
the strip when it returns.

### L1. Promised fallback that does not happen

`workers/streaming_classifier.py:644-649` logs "Falling back to standard
ASPRS classification" if the extra-byte dimension cannot be added, but pass
4 still writes to `field_name` (lines 757-758) and fails with a laspy
error. Raise a clear error instead of promising a fallback that is not
implemented (and per the project rule, should not be).

### L2. Dead branch

`workers/streaming_classifier.py:750-754`: both arms set `cls_dtype = np.uint8`.

### L3. Duplicate point listing in the block partition

`core/classifier_core.py:49-52`: for points in the first block along an
axis, `idx0 - 1` clips to `0`, so the same block index appears twice and the
point is listed twice in that block's group. Harmless (same voxel, same
prediction written twice) but wastes memory and time on every tile edge.

### L4. Model rebuilt and weights reloaded for every tile (measured: minor)

`filterPoints()` constructs `Segformer` and calls `torch.load` on each call
(`core/classifier_core.py:137-173`), and the tiled and streaming paths call
it once per tile. Measured in the venv on this machine (RTX, torch 2.12
cu126): 0.53 s cold, 0.12-0.14 s warm per call. Not a real cost today, but
it is the wrong shape for v1.1: the backend interface should load once per
run and predict per tile (section 4).

### L5. Model-specific strings scattered around

"AI Classification (3D SegFormer / TreeAIBox)" is hard-coded in three
places (`classifier_task.py:481`, `streaming_classifier.py:638`,
`classify_algorithm.py:507`); the class table is duplicated in the
Processing help text, the About dialog and `metadata.txt`. The
`TILE_AUTO_TARGET_POINTS` comment (`config.py:23-25`) cites GPU memory, but
the per-block tensor is fixed at 112 x 112 x 256 voxels regardless of tile
size, so the target only bounds host RAM.

### L6. Repository hygiene

The root of `QGIS_Plugin/` is the flat clone of the GitHub repo. It carries
untracked leftovers of the pre-QGIS "WaLoD2" desktop app (`walod2_main.py`,
`styles.py`, `gui/main_window.py`, `dialogs/about.py`,
`dialogs/class_manager.py`, `dialogs/class_remap.py`,
`workers/classifier_worker.py`, `model/`, `requirements.txt`,
`generate_icon.py`, preview PNGs) and `Extension/Benchmark` holds 12 GB of
LAS that is not ignored. A `git add -A` at the root would stage all of it.
Either delete the leftovers or add them and `Extension/` to `.gitignore`.

---

## 3. What is in good shape

Worth saying so it is not refactored away: streaming pass 4 (fresh output
record per chunk, dimension copy by name, PRF 6 / LAS 1.4 upgrade, COPC VLR
stripping, raw-byte EOF recovery with the point count made visible); the
input/output collision guard; SHA-256 verification with atomic rename and
mirror fallback; cancellation checked inside progress callbacks; the
settings migration for the GPU checkbox; the install-schema marker that
avoids gratuitous reinstalls; QgsTask instead of QThread for the run.

---

## 4. Multi-model readiness of the pipeline and UI

Everything assumes exactly one model:

| where | single-model assumption |
|---|---|
| `config.py` | `MODEL_INFO`, `MODEL_FILENAME`, `DEFAULT_MODEL_URL`, `MODEL_FALLBACK_URLS`, `MODEL_SHA256`, `MODEL_CONFIG_FILENAME`, `DEFAULT_CLASS_MAPPING` |
| `utils/model_manager.py` | all static methods, one path, one hash |
| `core/classifier_core.py` | `filterPoints` = load + voxelise + infer + argmax; XYZ only; returns ids 1-7 |
| `workers/classifier_task.py`, `streaming_classifier.py`, `processing/classify_algorithm.py` | `(config_path, model_path, class_mapping)` passed around; the "ids -> ASPRS, warn on unmapped" block and the write block exist three times |
| `gui/main_panel.py` | model label + download button for one model; `class_mapping = deepcopy(DEFAULT_CLASS_MAPPING)`; no model key in settings |
| `processing/classify_algorithm.py` | no `MODEL` parameter; class table baked into the help text |

Proposed shape for v1.1 (independent of which model comes second):

1. **Model registry** (`core/models/registry.py`): a `ModelSpec` dataclass
   with `id`, `display_name`, `family`, `version`, weights (filename,
   URLs, SHA-256, size), config file, `class_mapping` (model id ->
   `ClassInfo`), `required_dims` and `optional_dims` (LAS dimension names),
   resolution/voxel info, licence and attribution text, VRAM hint,
   `backend` id, and extra pip requirements. The registry lists the specs;
   the default stays the XYZ-only eSegFormer3D so "drop a file and it
   works" holds for first-time users.
2. **Backend interface** (`core/backends/base.py`): `load(spec, device)`
   once per run, `predict(xyz, features) -> model ids` per tile,
   `unload()`. `SegFormer3DBackend` wraps today's `filterPoints` minus the
   loading. Tiled and streaming paths call `backend.predict`.
3. **Pre-flight dimension check** (`utils/las_utils.py`): compare the input
   header's dimensions with `spec.required_dims`; the dock shows
   "continue / cancel / switch model", the Processing algorithm raises
   unless an explicit flag is set. Never zero-fill silently.
4. `ModelManager(spec)` instances instead of statics; weights cached under
   `.../models/<model_id>/`.
5. One shared `write_classified()` used by the task, the Processing
   algorithm and streaming pass 4 (removes the three copies).
6. Dock: model combo in the status strip with per-model status and
   download; persist `model_id` in `QgsSettings`. Processing: a `MODEL`
   enum parameter with stable ids so scripts and models keep working.
7. Pass `device: str` instead of `use_cuda: bool` end to end (fixes H2).
8. Runtime decision deferred until the model is known: keep in-process
   (dependencies must live with QGIS's Python and numpy) or add a
   subprocess runner inside the venv (true isolation; required if the model
   pins another torch/numpy or needs compiled extensions). The `backend`
   field in the spec leaves both open.

---

## 5. Install subsystem (venv, uv, CUDA cascade, downloads)

Reviewed separately: `utils/venv_manager.py` (~3 800 lines),
`utils/python_manager.py`, `utils/uv_manager.py`,
`workers/deps_install_worker.py`, `dialogs/deps_install_dialog.py`,
`dialogs/model_download_dialog.py`. Prefixes: V = venv_manager.py,
P = python_manager.py, U = uv_manager.py, W = deps_install_worker.py,
D = deps_install_dialog.py, M = model_download_dialog.py, MM =
model_manager.py, PL = plugin.py. I1 and I2 were re-verified by execution on
this machine; I4 is confirmed by the plugin's own 1.0.1 changelog; the
others come from reading and should be confirmed while fixing.

How it works: the gate `PL:180 -> V:3415 get_venv_status` is filesystem
only (standalone Python present, venv present, JSON marker
`install_schema_version == "2"`, torch/torchvision directories). Install:
`PL:253` -> `detect_nvidia_gpu` (nvidia-smi, cached for the session) ->
`DepsInstallWorker` (QThread) wipes `VENV_DIR` -> `V:3544
create_venv_and_install`: python-build-standalone 20241219 (P:232) and uv
0.10.6 (U:99) downloaded through `QgsBlockingNetworkRequest`, `python -m
venv [--copies on Windows] --without-pip` (V:1509), then `V:2361
install_dependencies`: Phase A torch + torchvision from
`download.pytorch.org/whl/<cuXXX>` with the version cap, Phase B the rest
from PyPI, marker (V:3364), `verify_venv` (imports each package in the
venv Python, V:3109), CUDA smoke test `torch.zeros(1, device="cuda")`
(V:2254), and on failure the cascade cu128 -> cu126 -> cu124 -> cu121 ->
cu118 (V:2119, V:2159). At run time `ensure_venv_packages_available`
(V:1186) puts the venv site-packages on QGIS's `sys.path` (plus
`os.add_dll_directory` on Windows) and torch is imported into the QGIS
process.

### I1 (High, verified). The plugin does not load on Python 3.9 to 3.11

Four files use single-quoted f-strings with a line break inside `{...}`,
which is PEP 701 syntax, Python 3.12 only: `V:140`, `V:154`, `V:1730`,
`P:71`, `P:295`, `P:302`, `gui/main_panel.py:1244`,
`workers/streaming_classifier.py:555`. Compiling the plugin with Python
3.11.5 fails in all four files ("unterminated string literal").
`metadata.txt` says `qgisMinimumVersion=3.34`; QGIS 3.34 ships Python 3.9
in the official macOS builds, and the Ubuntu 22.04 / Debian 12 packages
use Python 3.10 / 3.11. On those, opening the dock or the Setup panel
dies with a SyntaxError. The strings look like an autopep8 artefact (see
also `V:137-138`). Fix: rejoin the strings; add a "compile with Python
3.9" step to the release checklist. Follow-on to verify once fixed:
`_safe_extract_tar` (P:76-78) refuses symlinks on < 3.12, and the
python-build-standalone Linux/macOS tarballs contain them
(`bin/python3 -> python3.X`).

### I2 (High, verified). Phase B replaces the CUDA torch with the CPU torch from PyPI

`V:2828-2838` runs `uv pip install --upgrade laspy lazrs timm numpy_indexed
"numpy<2.0"` with no index URL after Phase A. `timm` depends on torch and
torchvision, and `--upgrade` makes uv re-resolve every package in the
graph, so torch comes from PyPI at its latest version. Dry-run of that
exact command against the real venv on this machine (2026-09-25):
`- torch==2.12.0+cu126  + torch==2.14.0`,
`- torchvision==0.27.0+cu126  + torchvision==0.29.0`. On Windows the PyPI
torch is CPU-only, so the smoke test fails and the cascade reinstalls at
cu124, cu121 or cu118 (an older toolkit than the driver supports, after
extra multi-GB downloads); on Linux the PyPI torch drags in ~3 GB of
`nvidia-*` wheels. It only worked in May because PyPI's latest version
equalled the capped `+cu126` one; every torch release since makes it
worse, and the cap table (V:99-108, "as of May 2026") ages the same way.
Fix: drop `--upgrade` (the venv is always freshly wiped, W:63-69) or use
`--upgrade-package` for the named packages only, and pass a constraints
file pinning torch and torchvision to the Phase A versions to every later
install (Phase B, `_reinstall_cpu_torch` V:2069, the cascade). Persist
the resolved torch version and `cuda_index` in the marker.

### I3 (High). The CUDA cascade can leave the venv without torch and still report "ready"

`_reinstall_torch_at_cuda_index` uninstalls torch + torchvision first
(V:2185-2195); if the install then fails it returns False and the loop
continues (V:3796-3799). `is_valid` (V:3731) is never recomputed, the
marker was already written (V:3051), the function returns True (V:3838)
and the `cuda` flag is written despite the failed smoke test (V:3823).
The dock opens, the torch import fails, and only the next launch says
"incomplete". Fix: re-run `verify_venv` after the cascade, reinstall the
original index's torch on exhaustion, write the marker only after
verification succeeds.

### I4 (High, security). TLS verification is off for every wheel download

`_get_uv_ssl_flags` (V:640-671) adds `--allow-insecure-host` for pypi.org,
files.pythonhosted.org and download.pytorch.org on the first attempt
(V:2081, 2211, 2509, 2713, 2836, 2946; pip path `--trusted-host`
V:619-637). uv documents that flag as disabling certificate checks, and
no wheel hashes are pinned, so anyone on the network path can serve
arbitrary wheels that QGIS then imports in-process. The docstring calls
it a "last-resort fallback"; it is the default. It also makes the SSL
error branches (V:2774, V:2998) unreachable for uv, while the python, uv
and model downloads still go through Qt with verification, so a corporate
inspection proxy fails at step 1 regardless. Fix: first attempt with
`--native-tls` only; add `--allow-insecure-host` only on a retry after
`_is_ssl_error`, with a logged warning (and ideally a user confirmation).

### I5 (Medium). One failed nvidia-smi call pins the user to CPU torch

V:446-452 caches `(False, {})` for the session after a 5 s timeout
(Optimus laptops, battery mode, first driver load); PL:261 turns that into
`cuda_enabled=False`; the marker and the `cpu` flag are written;
`get_venv_status` never compares the installed torch with the GPU, and
`_read_cuda_flag` (V:217) is never called anywhere. Fix: do not cache
timeouts or errors; compare `_read_cuda_flag()` with `detect_nvidia_gpu()`
in `get_venv_status` and offer an "Enable GPU" action.

### I6 (Medium). A partially extracted standalone Python is never repaired

P:309-320 rmtrees then extracts; on failure only the archive is removed
(P:358-363), and on verification failure the directory is kept
(P:331-332). `standalone_python_exists()` (P:148-155) checks only
`python.exe`, and `create_venv_and_install` (V:3602) then skips both the
download and the verification. AV quarantine of one DLL mid-extract means
every later install uses a broken interpreter behind a misleading venv
error. Fix: extract to a temp dir and rename atomically; rmtree on any
failure; re-verify at V:3602.

### I7 (Medium). Cancel is cosmetic; `terminate()` orphans the uv child

PL:310-314 flips the dock back to the Install state immediately, but
cancel is only polled in `_run_pip_install` (V:1875) and between phases,
never inside `create_venv` (V:1622, 120 s), `_verify_cuda_in_venv`
(V:2282, up to 300 s), the cascade reinstall (V:2230, 900 s x 2),
`_reinstall_cpu_torch` (V:2098), `verify_venv`, or the blocking Qt
downloads. A second Install click is silently dropped (PL:255-256).
`unload()` (PL:106-113) calls `QThread.terminate()`: the uv child keeps
writing into the venv and the `finally` cleanup of `_run_pip_install`
(V:1960-1978) never runs. Fix: keep the `Popen` on the worker and kill it
from `cancel()`; never `terminate()`; disable Install until `finished`;
route the four `subprocess.run` sites through `_run_pip_install`.

### I8 (Medium). Model download dialog: Cancel does not cancel; unload mid-download aborts QGIS

M:82-84 wires Cancel to `reject` and never disables it; `DownloadWorker`
(M:15-30) has no parent and no cancel. The dialog is parented to the dock,
so unloading the plugin during a download destroys a running QThread (Qt
fatal). Progress is decorative: MM:240-243 emits one `(total, total)`
after the whole body is already in memory. Fix: disable or implement
Cancel, parent the worker and `wait()` in `closeEvent`; for a larger
second model, stream to the `.part` file (`QgsFileDownloader`) with real
progress.

### I9 (Low). Reinstall runs a multi-GB `rmtree` on the UI thread

D:215-224 calls `remove_venv()` synchronously before emitting
`install_requested`, then the worker rmtrees again (W:63-69). QGIS
freezes; a Windows `PermissionError` is swallowed. Fix: delete D:217-222.

### I10 (Low). Proxy handling is HTTP-basic only

`_get_qgis_proxy_settings` (V:1039-1074) ignores `proxy/proxyType` (a
SOCKS5 proxy gets an `http://` URL) and `proxy/proxyExcludedUrls` (no
`NO_PROXY`); QGIS auth-config proxies are not passed to uv at all.

### I11 (Low). The `numpy<2.0` pin does not govern what runs

V:42-44 pins numpy in the venv, but torch, laspy and lazrs are imported
in-process and use QGIS's own numpy (`classifier_task.py:12` imports numpy
before the venv path is added; QGIS 3.44 bundles numpy 2.x). The pin costs
a download and documents a constraint that is not enforced. Verify lazrs
and laspy against the host numpy (import in-process) instead.

### Extensibility for a second model (install side)

- One flat `REQUIRED_PACKAGES` (V:30-48); `_get_required_packages()`
  (V:250-256) is the only hook and takes no arguments. torch and
  torchvision are special-cased by name in nine places (V:2410,
  2450-2453, 2476-2482, 2069, 2201, 3074-3077, 3088-3091, 3328-3331,
  1228-1232).
- `_get_verification_code` (V:3081-3106) guesses the import name from the
  distribution name; any dependency where they differ (`scikit-learn` ->
  `sklearn`, `Pillow` -> `PIL`) fails verification and aborts the install
  (V:3173). Package entries need an explicit `import_name`.
- No incremental install: the worker always wipes. Adding a package for
  model 2 does nothing for existing users unless `_INSTALL_SCHEMA_VERSION`
  is bumped, which forces a full 2 to 5 GB rebuild. Phase B (V:2811-3036)
  is close to reusable as `_install_batch(specs, constraints)`.
- The marker is already JSON (V:3387-3392): add `packages`, `cuda_index`,
  `torch_version`, `models` so `get_venv_status` can compute what is
  missing. `cuda_index` is a loop-local today (V:2459) and never
  persisted; any torch-keyed extra (torch-scatter, torch_geometric) needs
  it.
- One venv per Python version (V:28); `cleanup_old_venv_directories`
  (V:3267-3279) deletes sibling `venv_py*`. A second model must share the
  torch build: one `TORCH_SPEC` for all models plus per-model
  `extra_packages` installed incrementally under a torch constraint (see
  I2). A model that pins a different torch is not feasible in this design
  without a subprocess runner and a second venv.
- Model side: constants in `config.py`, static `ModelManager`, one
  `model_url` settings key, a dialog and a panel that hard-code
  `MODEL_INFO['name']`. Same `ModelSpec` refactor as section 4.

### Quick wins

- `_select_cuda_index` (V:455-539) duplicates `_get_cuda_cascade_candidates`
  (V:2119-2156): `c = _get_cuda_cascade_candidates(g); return c[0] if c else None`.
- Six near-identical install command builders (V:2069, 2207, 2501, 2705,
  2828, 2938) and duplicated retry / classification chains (V:2597-2650 vs
  2871-2922; 2769-2809 vs 2988-3028). `install_dependencies` is ~700
  lines (V:2361-3057), `create_venv_and_install` ~300 (V:3544-3839).
- The pip (non-uv) path is dead on Linux by its own comment (V:1582-1589
  then V:1666-1708) and untested elsewhere; making uv mandatory removes
  ~200 lines (`_get_pip_ssl_flags`, `_get_pip_proxy_args`,
  `get_venv_pip_path`, every `else:` branch).
- Dead code: `_read_cuda_flag` (V:217), `_INSTALL_LOGIC_VERSION` (V:54)
  and the deps-hash machinery (V:239-279, drift ignored at 3501-3508),
  `_CUDA_LOGIC_VERSION` (V:57, only embedded in strings), the `requests`
  branch of `_get_verification_code` (V:3102), `_is_optional_verify_package`
  / `_is_optional_install_package` (V:2342-2358) and the `sam3` branch
  (V:2326) that reference packages not in `REQUIRED_PACKAGES`, so the
  "retry without optional package" block (V:2925-2982) is unreachable;
  `_force_cuda_reinstall` logs a `--force-reinstall` that is never passed
  (V:2432-2436 vs 2500-2521).
- Sentinel tags `[CUDA_FALLBACK]`, `[DRIVER_TOO_OLD]`, `[CUDA_VERIFY_FAILED]`
  are shown raw to the user (PL:277 -> D:284) and parsed by substring
  (V:3722-3723); return a small result object instead.
- The schema comment (V:67-70) was not updated for the v1.0.2 venv change
  (commit e3eacfc) despite the rule at V:59-62.
- `_get_subprocess_kwargs` (V:1028) does `os.makedirs` on every call;
  `cleanup_old_venv_directories` (V:3261) deletes venvs that belong to
  another QGIS install's Python sharing the same cache dir.

---

## 6. Suggested order of work

1. Self-contained fixes, in this order: H1 (done), I1 (Python 3.9-3.11
   syntax), I2 (Phase B `--upgrade`), I4 (TLS), I3 (cascade state), H2,
   H3, M1-M4, I5-I9. H1 + I1 + I2 together justify a 1.0.3 patch release
   before 1.1.0: today, flat terrain gets wrong output, Python 3.9-3.11
   QGIS builds cannot open the plugin, and every fresh Windows install
   ends up on CPU torch (or on an older CUDA toolkit after extra
   downloads).
2. Refactor per section 4 (registry, backend, shared writer, pre-flight
   check, device string) plus the install side of section 5 (package specs
   with import names, incremental install under a torch constraint, JSON
   marker with packages / cuda_index / torch_version), with the current
   model as the only entry.
3. Integrate the second model when it is provided: write its `ModelSpec`,
   its backend, its dependency set, and run the benchmark tiles in
   `Extension/Benchmark` through both models.
