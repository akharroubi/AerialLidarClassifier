# Aerial LiDAR Classifier

<p align="center"><img src="assets/logo_128.png" alt="Aerial LiDAR Classifier logo" width="128" height="128"></p>

Deep-learning semantic segmentation of aerial LiDAR point clouds (LAS / LAZ / COPC)
inside QGIS, with two models:

- **LitePT-L** (default on NVIDIA GPUs): a point transformer from
  [prs-eth/LitePT](https://github.com/prs-eth/LitePT) trained on the
  [DALES](https://arxiv.org/abs/2004.11985) aerial LiDAR dataset at 10 cm.
  Eight classes, custom four-tile DALES test mIoU 0.824.
- **3D SegFormer** (UrbanFiltering, [TreeAIBox](https://github.com/NRCan/TreeAIBox),
  Natural Resources Canada): a voxel transformer at 30 cm that also runs on CPU and
  Apple Silicon.

[![QGIS](https://img.shields.io/badge/QGIS-3.34%2B-1f9b4e)](https://qgis.org)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-3776AB)](https://www.python.org)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![Model: CC BY-NC 4.0](https://img.shields.io/badge/model-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-555)](#hardware-and-platform-support)

The plugin produces ASPRS-compliant LAS/LAZ output: every input point is written
back to the standard `classification` dimension with codes that follow the
[ASPRS LAS 1.4](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)
specification, so downstream tools (PDAL, LASTools, CloudCompare, Potree, QGIS
PDAL provider) read the result with no special handling.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Hardware and platform support](#hardware-and-platform-support)
- [Classification output](#classification-output)
- [Coordinate units](#coordinate-units)
- [Tiling and streaming](#tiling-and-streaming)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Model weights](#model-weights)
- [Citation](#citation)
- [License](#license)
- [Credits](#credits)

---

## Features

- **Two models, one interface.** Pick LitePT-L or SegFormer 3D in the dock or
  with the `MODEL` parameter in Processing; each model says which devices it
  runs on, and its weights are downloaded (or imported from a file) once.
- **Native dock-based UI** with drag-and-drop, layer picker, and live log.
- **Processing algorithm and provider** &mdash; usable from the Toolbox, the
  Graphical Modeler, and the `qgis_process` command-line interface.
- **Isolated dependency installer.** On first launch the plugin downloads a
  portable Python and the required wheels into a per-user virtual environment
  under `~/.qgis_aerial_lidar_classifier/`. The system Python and QGIS's own
  bundled Python are never touched. No admin rights required.
- **Automatic CUDA wheel selection.** NVIDIA driver and compute-capability are
  detected via `nvidia-smi`; the matching PyTorch wheel
  (`cu118` / `cu121` / `cu124` / `cu126` / `cu128`, cu126 preferred) is
  installed, plus `spconv` for LitePT-L. Explicit CPU installation is available for SegFormer; failed CUDA installs do not silently fall back.
  Apple Silicon uses MPS (SegFormer 3D only).
- **Units handled.** Feet are converted from the file's CRS before inference;
  see [Coordinate units](#coordinate-units).
- **Spatial tiling with halo** for files larger than GPU memory.
- **Streaming I/O** (4-pass: header &rarr; per-tile partition &rarr; per-tile
  inference &rarr; streaming output write) so files larger than RAM can still
  be processed.
- **SHA-256-verified model download** through `QgsBlockingNetworkRequest` &mdash;
  QGIS proxy, authentication, and certificate settings are honoured.
- **Background `QgsTask` execution** with progress and cancellation.
- **Optional auto-load** of the classified result as a QGIS point-cloud layer.

---

## Installation

### From the QGIS plugin repository (recommended)

1. *Plugins &rarr; Manage and Install Plugins...*
2. Search for **Aerial LiDAR Classifier**.
3. Click **Install Plugin**.

### From source

```bash
git clone https://github.com/akharroubi/AerialLidarClassifier.git
```

Copy the cloned folder into your QGIS plugins directory:

| OS      | Path                                                                            |
|---------|---------------------------------------------------------------------------------|
| Windows | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\AerialLidarClassifier\`   |
| macOS   | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/AerialLidarClassifier/` |
| Linux   | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/AerialLidarClassifier/` |

Then restart QGIS and enable the plugin from the plugin manager.

### First launch

The first time you open the dock or run the Processing algorithm, the plugin
opens a **Setup** panel that downloads:

- a standalone Python interpreter (~30&nbsp;MB),
- [`uv`](https://github.com/astral-sh/uv) (single static binary),
- PyTorch + supporting wheels (~1&ndash;3&nbsp;GB depending on CUDA), and on
  CUDA installs `spconv` (sparse convolutions for LitePT-L, ~100&nbsp;MB),
- the model weights when you first select a model: LitePT-L ~172&nbsp;MB,
  SegFormer 3D ~18&nbsp;MB (both SHA-256 verified; a weights file you already
  have can be imported from the dock instead).

Total install time on a residential connection is typically 3&ndash;10&nbsp;minutes.
Everything is stored under a per-user folder and can be removed at any time
without affecting QGIS.

---

## Quick start

1. Open the dock from the toolbar (or *Plugins &rarr; Aerial LiDAR Classifier*).
2. Drag a `.las` / `.laz` / `.copc.laz` file into the input list, **or** select
   an already-loaded point-cloud layer from the dropdown.
3. Choose an output folder.
4. Click **Run**.

The output is LAS or LAZ (COPC input becomes ordinary LAZ, without a COPC
index). Point order, coordinates, scaling, CRS, VLRs/EVLRs and unselected
attributes are retained. Standard `classification` uses ASPRS codes; a custom
integer label field retains the raw model IDs. A model-provenance VLR is added.
Legacy point formats are upgraded only when a class code requires it.

---

## Usage

### Dock panel

The dock is organised as two stacked groups:

- **Input** &mdash; drop list, "Add files...", "Add folder...", layer picker.
- **Output** &mdash; folder, suffix (default `_classified`), format,
  classification field (default `classification`), and an "Auto-load result as
  layer" toggle.

Advanced parameters (collapsible) expose:

- **Compute device** &mdash; Use GPU checkbox (CUDA/MPS when available);
  Processing additionally exposes Auto, GPU and CPU choices.
- **Tiling** &mdash; enable, tile size (m), buffer (m), streaming I/O.

The bottom pane is always-visible log output with timestamps and severity
colouring; it mirrors the entries written to the QGIS Log Messages panel under
the *Aerial LiDAR Classifier* tag.

### Processing algorithm

The plugin registers a Processing provider, so the algorithm
`aeriallidar:classify_lidar` is available from:

- the **Processing Toolbox**,
- the **Graphical Modeler** (chainable with any other Processing alg; the
  classified file is exposed as the `OUTPUT_FILE` output), and
- the `qgis_process` CLI:

```bash
qgis_process run aeriallidar:classify_lidar \
  --INPUT=/data/tile.laz \
  --OUTPUT_FOLDER=/data/classified \
  --TILE_ENABLED=true \
  --TILE_SIZE_M=200 \
  --TILE_BUFFER_M=20
```

All UI options are exposed as algorithm parameters; advanced options
(`FIELD_NAME`, `TILE_STREAMING`, `UNITS`, &hellip;) live under the Processing
dialog's *Advanced parameters* group.

---

## Hardware and platform support

| Environment | Validation scope |
|-------------|------------------|
| Windows 10, QGIS 3.44.10, Python 3.12, RTX 3090 | QGIS I/O, CPU and CUDA inference, installer and regression tests |
| Linux x86_64, Ubuntu under WSL, Python 3.10 | Installer/runtime tests; QGIS download/settings shim, not Linux QGIS GUI |
| macOS / Apple MPS | Implementation present; not validated for this release |
| QGIS 4 / Qt6 | Not declared supported in this release |
| Other GPU generations, small physical GPUs, Python 3.13 | Not validated on hardware/runtime combinations in this release |

**LitePT-L requires NVIDIA CUDA and compatible spconv.** This installer's
supported spconv indexes are cu118, cu121, cu124 and cu126; LitePT remains
unavailable on a cu128 installation. Default crops contain up to 70,000 points.
The small-card path uses 35,000-point / 20 m crops; OOM retries reduce the point
budget down to 5,000. These changes can affect labels. Neither crop size nor
streaming is a hard total VRAM limit: native library workspaces and the QGIS
renderer also use memory. Rates and memory measurements are dataset-specific.

**SegFormer 3D** supports CPU and CUDA; an Apple MPS path is present but was
not tested for this release. CPU mode keeps inference off CUDA but can take
substantially longer and still uses CPU and RAM.

The right CUDA wheel is selected automatically from `nvidia-smi`'s reported
compute capability *and* driver version; you do not need to install CUDA
yourself.

---

## Classification output

Output points use the ASPRS LAS 1.4 classification codes.

**LitePT-L** (DALES classes):

| Model class | Name        | ASPRS code | Meaning                  |
|------------:|-------------|-----------:|--------------------------|
| 1           | Ground      | 2          | Ground                   |
| 2           | Vegetation  | 5          | High Vegetation          |
| 3           | Cars        | 1          | Unclassified&nbsp;\*     |
| 4           | Trucks      | 1          | Unclassified&nbsp;\*     |
| 5           | Power lines | 14         | Wire &mdash; Conductor   |
| 6           | Fences      | 1          | Unclassified&nbsp;\*     |
| 7           | Poles       | 15         | Transmission Tower       |
| 8           | Buildings   | 6          | Building                 |

**SegFormer 3D** (UrbanFiltering classes):

| Model class | Name        | ASPRS code | Meaning                  |
|------------:|-------------|-----------:|--------------------------|
| 1           | Ground      | 2          | Ground                   |
| 2           | Vegetation  | 5          | High Vegetation          |
| 3           | Vehicles    | 1          | Unclassified&nbsp;\*     |
| 4           | Wiring      | 14         | Wire &mdash; Conductor   |
| 5           | Fence       | 1          | Unclassified&nbsp;\*     |
| 6           | Pole        | 15         | Transmission Tower       |
| 7           | Building    | 6          | Building                 |

\* ASPRS LAS 1.4 has no dedicated codes for vehicles or fences, so these are
mapped to **1 = Unclassified**. If you need them as separate classes, set
`FIELD_NAME` to another name to write the raw model ids into an extra LAS
dimension instead.

When the input file already uses Point Record Format &lt; 6, the writer
upgrades it to PRF 6 / LAS 1.4 (required for ASPRS codes &gt; 31). All other
attributes (RGB, intensity, return number, GPS time, scan angle, extra
dimensions) are preserved byte-for-byte.

---

## Coordinate units

The model works in **metres**. Most US LiDAR is delivered in US survey feet
(state plane coordinate systems); fed to the model unconverted, every distance
is 3.28 times too small for it, buildings come out as *Wire - Conductor* and
*Transmission Tower*, and the run is about 13 times slower because the model
sees ten times more voxel blocks.

Since 1.0.3 the plugin reads the linear unit from the file's CRS (the WKT
record of LAS 1.4 files, or the GeoTIFF keys of older files), converts XY and
Z to metres **for the model only**, and logs what it found:

```
tile.las: coordinate units: XY in US survey foot, Z in US survey foot
(WKT CRS: US survey foot horizontal, US survey foot vertical); converting to metres for the model
```

The output file keeps the original coordinates, scales and CRS untouched.

- When the header carries no CRS, metres are assumed and the log says so.
- When the CRS has no vertical component, Z is assumed to share the
  horizontal unit (the usual convention for US deliveries).
- Geographic files (degrees) are refused with a message: reproject them first
  (for example to the local UTM zone).

To force a unit when the header is missing or wrong, use *Advanced parameters
&rarr; Input units* in the dock (auto-detect, metres, international feet, US
survey feet) or the `UNITS` parameter of the Processing algorithm.

---

## Tiling and streaming

For files larger than your GPU memory, enable **Tiling** in the advanced
parameters:

- **Tile size (m)** &mdash; lateral extent of each tile in the same units as
  the file CRS. Default `200`.
- **Buffer (m)** &mdash; halo around each tile providing spatial context for
  edge points. Default `50`. Points inside the buffer are inferred but not
  written; only the core tile is.

For files **larger than RAM**, additionally enable **Streaming I/O**. The
classifier then runs in four passes:

1. Read the LAS header to compute bounding box and point count.
2. Partition points into per-tile NumPy `.npz` sidecars on disk.
3. Run inference on one tile at a time (only one tile's points are in memory).
4. Stream-write the output LAS/LAZ file as classified tiles complete.

Streaming requires tiling to be enabled. The UI couples the two checkboxes
automatically.

---

## How it works

**LitePT-L** works on points, not voxels. Per tile the plugin removes an
integer origin, keeps one representative per occupied 10&nbsp;cm voxel (with an
exact membership map back to every raw point), and covers the representatives
with crops of at most 70&nbsp;000 points within 30&nbsp;m (regular centres, then one
crop per still-uncovered point). Each crop is centred, floor-referenced and
voxelised exactly like a validation crop, the network's softmax is averaged
over every visit of a point, and the argmax is projected back to the raw
points. The upstream code (MIT) is vendored in `core/litept/` with pure-PyTorch
replacements for FlashAttention, `torch_scatter` and the RoPE kernel, so only
`spconv` remains compiled. This port was checked against the reference
implementation: 99.998&nbsp;% of 2.7&nbsp;M points identical.

**SegFormer 3D** is the **UrbanFiltering 3D SegFormer**
([Xi *et&nbsp;al.*, TreeAIBox](https://github.com/NRCan/TreeAIBox)), a hierarchical
transformer adapted to volumetric point-cloud features. The plugin's inference
pipeline:

1. **Voxelisation** &mdash; points are projected into a regular 30&nbsp;cm voxel
   grid; each occupied voxel carries height, intensity, and density features.
2. **Patch extraction** &mdash; the grid is split into overlapping
   112&times;112&times;48 patches.
3. **Forward pass** &mdash; SegFormer 3D produces per-voxel logits (7 classes).
4. **De-voxelisation** &mdash; voxel labels are scattered back to the original
   points via nearest-voxel lookup, so the output preserves the input point
   density exactly.

The whole pipeline runs in a `QgsTask` so it never blocks the QGIS event loop.
Progress (percent + status string) is reported back through standard
`QgsTask` signals.

---

## Troubleshooting

Every install / dependency / detection step writes to the QGIS Log Messages
panel under the **Aerial LiDAR Classifier** tag. If something fails, open
*View &rarr; Panels &rarr; Log Messages* first &mdash; the answer is almost
always there.

### Install fails with `WinError 4551` / "Application Control policy" / AppLocker / WDAC

This is **not** antivirus &mdash; it's Windows Application Control (AppLocker,
WDAC, or Smart App Control), a kernel-level allow-listing policy enforced by
IT. It refuses to execute *any* binary not on the corporate allow-list, and
the user cannot bypass it. Symptoms:

- Setup dialog shows `Failed to create venv: [WinError 4551] ...`, or the
  French equivalent `Une stratégie de contrôle d'application a bloqué ce
  fichier`.
- AV exclusions do nothing.
- Cache redirect via `AERIAL_LIDAR_CLASSIFIER_CACHE_DIR` does nothing
  (Application Control is per-binary, not per-folder).
- Running QGIS as administrator does nothing if the policy is machine-wide.

**The only fix is IT involvement.** Send this exact request to your IT
team:

> Please allow execution under `C:\Users\<my-username>\.qgis_aerial_lidar_classifier\`
> for my user account &mdash; either by adding the folder to the AppLocker /
> WDAC allow list, or by signing the python-build-standalone binaries used
> by this QGIS plugin.

If your IT team will not grant this, the plugin cannot run on a machine
with that policy in force. Use a personal / unmanaged Windows machine, or
Linux / macOS, instead.

### Install fails with "blocked by antivirus / endpoint-security product"

By far the most common first-run failure on Windows laptops, especially
corporate / shared / managed machines. Defender or third-party AV scans the
freshly downloaded portable `python.exe` inside the venv and quarantines it
between creation and use. The plugin detects this and tells you the exact
folder to whitelist (typically `C:\Users\<you>\.qgis_aerial_lidar_classifier`).

**If you have admin rights** (Windows Defender, from an *elevated* PowerShell):

```powershell
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.qgis_aerial_lidar_classifier"
Remove-Item -Recurse -Force "$env:USERPROFILE\.qgis_aerial_lidar_classifier"
```

Then click *Reinstall Dependencies* in the Setup panel.

**If you don't have admin rights** (intern / managed laptop):

Either ask IT to add the exclusion above, or redirect the cache to a folder
that's already whitelisted by your organisation. The plugin honours the
`AERIAL_LIDAR_CLASSIFIER_CACHE_DIR` environment variable:

```powershell
[Environment]::SetEnvironmentVariable(
    "AERIAL_LIDAR_CLASSIFIER_CACHE_DIR",
    "C:\Dev\aerial_lidar_classifier",
    "User"
)
```

Restart QGIS afterwards. The Setup panel will use the new location.

### The Setup panel keeps appearing on every launch

The dependency check failed. Filter the Log Messages panel on the
*Aerial LiDAR Classifier* tag and inspect the lines starting with
`Dependency check:`. Most common causes:

- An interrupted first install &mdash; click *Reinstall* in the Setup panel.
- A previous install was quarantined (see AV section above).
- The venv lives somewhere QGIS no longer has read access to (e.g. you
  moved your user profile). Delete `~/.qgis_aerial_lidar_classifier/` and
  reinstall.

### "Failed to inspect Python interpreter" or "Failed to query Python interpreter"

These are `uv`'s error messages when the venv's `python.exe` was created
successfully but then disappeared or refused to start before `uv` could use
it. **99 % of the time this is antivirus quarantine** &mdash; see the first
section. The plugin recognises both phrases and routes them through the
AV-help path automatically.

### `nvidia-smi` was not found but I have an NVIDIA GPU

The plugin queries `nvidia-smi` to read driver version + compute capability.
On Windows the binary normally lives in `C:\Windows\System32\nvidia-smi.exe`
(already on `PATH`); the plugin also falls back to
`C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe`.

To check what's installed:

```powershell
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader,nounits
Get-ChildItem -Path "C:\Program Files\NVIDIA Corporation","C:\Windows\System32" -Filter "nvidia-smi.exe" -ErrorAction SilentlyContinue | Select-Object FullName
```

If the first command works in a normal PowerShell but the plugin still says
"GPU not detected", check the Log Messages panel &mdash; every detection
failure path is now logged with the reason.

If `nvidia-smi` itself fails, the NVIDIA driver is incomplete or stale.
Reinstall the **Game Ready** or **Studio** driver from
[nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
(not from Windows Update / OEM tools, which can ship a partial driver
without `nvidia-smi`).

### Almost everything is classified as wires or towers, or the run is very slow

The file is in feet. See [Coordinate units](#coordinate-units): versions up
to 1.0.2 fed the coordinates to the model unconverted. Update to 1.0.3 or
later, or set *Input units* explicitly if the file's CRS is missing or wrong.

### `SyntaxError: unterminated string literal (detected at line 140)` when opening the plugin

Versions 1.0.0 to 1.0.2 could not load on QGIS builds with Python 3.9 to
3.11 (Ubuntu 22.04, Debian 12, the official macOS 3.34 package). Fixed in
1.0.3; update from the plugin manager.

### Inference runs on CPU even though my GPU is detected

Open the dock, expand *Advanced parameters &rarr; Compute device*, and set
it to *CUDA* (or *MPS*) explicitly. Auto-selection only switches to GPU when
the detected device reports at least 2&nbsp;GB of free memory at start-up.

### `Could not open file as point cloud layer` when auto-loading the result

This is usually a missing PDAL provider in your QGIS build. Open the file
manually via *Layer &rarr; Add Layer &rarr; Add Point Cloud Layer* &mdash;
if that also fails, your QGIS install does not ship the PDAL provider.
Use a QGIS package that bundles PDAL (the official Windows / macOS
installers from qgis.org all do).

### The model download fails behind a corporate proxy

The plugin uses `QgsBlockingNetworkRequest`, which honours the proxy
configured in *Settings &rarr; Options &rarr; Network*. Verify those
settings and retry; the fallback URL in `config.py` is tried automatically.
If both URLs fail you can also drop the `.pth` file (~18&nbsp;MB) manually
into the models folder of your QGIS profile
(`%APPDATA%\QGIS\QGIS3\profiles\default\AerialLidarClassifier\models\` on
Windows, `~/.local/share/QGIS/QGIS3/profiles/default/AerialLidarClassifier/models/`
on Linux) &mdash; the plugin will verify its SHA-256 and use it without ever
touching the network.

### Install takes a long time / progress bar appears stuck

The PyTorch + CUDA wheels are 1&ndash;3&nbsp;GB depending on which CUDA
version was selected. On a slow connection the download can genuinely take
ten minutes or more. The Log Messages panel shows the live progress; if
you see new lines, the install is still working.

---

## Model weights

Weights are downloaded when a model is first selected (or imported from a file
with the folder icon next to the model selector).

**LitePT-L (DALES, 10 cm)**

| Field         | Value |
|---------------|-------|
| File          | `litept_l_dales_10cm_ema_fp16.pth` (EMA weights of the validation-selected checkpoint, stored as float16) |
| Size          | ~172&nbsp;MB |
| SHA-256       | `849ba5089e629785fd64f5166cc35f999b758c68754573bf18122a277b09592b` |
| URL           | [this repo, release v1.1.0 (tag v1.1)](https://github.com/akharroubi/AerialLidarClassifier/releases/tag/v1.1) |
| Training data | DALES (Dayton Annotated LiDAR Earth Scan), 32 tiles; validated on 4, tested on 4 held-out tiles |
| License       | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (maintainer-selected licence for the trained weights) |

**SegFormer 3D (UrbanFiltering, 30 cm)**

| Field         | Value |
|---------------|-------|
| File          | `urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth` |
| Size          | ~18&nbsp;MB |
| SHA-256       | `cddb791041d46a2e7be3c53d6e94157c9af2218e7ea5595bf3134c4390c1fbc0` |
| Primary URL   | [NRCan/TreeAIBox release v1.0](https://github.com/NRCan/TreeAIBox/releases/tag/v1.0) |
| Mirror URL    | [this repo, release v1.0.0](https://github.com/akharroubi/AerialLidarClassifier/releases/tag/v1.0.0) |
| License       | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

Both URLs return the byte-identical file; the SHA-256 above is verified after
every download and a mismatch is treated as a hard failure.

**Commercial use of the model weights requires a separate licence from the
model author** (see *Credits* below). The plugin source code itself is under
GPL-3.0-or-later and is free to use commercially.

---

## Citation

If you use this plugin in academic work, please cite the model you used.
For LitePT-L, cite the LitePT architecture (see the
[LitePT repository](https://github.com/prs-eth/LitePT) for the paper) and the
DALES dataset it was trained on:

```bibtex
@inproceedings{varney2020dales,
  author    = {Varney, Nina and Asari, Vijayan K. and Graehling, Quinn},
  title     = {{DALES}: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.11985}
}
```

For SegFormer 3D:

```bibtex
@misc{xi_treeaibox,
  author       = {Xi, Zhouxin},
  title        = {TreeAIBox: deep learning for forestry and urban LiDAR},
  howpublished = {\url{https://github.com/NRCan/TreeAIBox}},
  organization = {Natural Resources Canada},
  note         = {CC BY-NC 4.0}
}
```

And, optionally, the plugin itself:

```bibtex
@software{kharroubi_aerial_lidar_classifier,
  author       = {Kharroubi, Abderrazzaq},
  title        = {{Aerial LiDAR Classifier}: a QGIS plugin for deep-learning
                  semantic segmentation of aerial LiDAR point clouds},
  year         = {2026},
  url          = {https://github.com/akharroubi/AerialLidarClassifier},
  version      = {1.1.0},
  note         = {GPL-3.0-or-later}
}
```

---

## License

- **Plugin source code** &mdash; [GPL-3.0-or-later](LICENSE). The vendored
  LitePT model code (`core/litept/`) is MIT, &copy; Photogrammetry and Remote
  Sensing Lab, ETH Zurich (see `core/litept/LICENSE.upstream`).
- **Model weights** &mdash; [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  LitePT-L weights were trained by the plugin maintainer on DALES; this
  weight licence is not inferred automatically from the dataset licence.
  SegFormer weights are distributed unchanged from TreeAIBox. The maintainer
  confirmed author permission for SegFormer integration. See
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and scope.
  Commercial users must obtain a separate licence from the model authors.

---

## Credits

**LitePT-L.** Architecture and reference implementation by the Photogrammetry
and Remote Sensing Lab, ETH Zurich ([prs-eth/LitePT](https://github.com/prs-eth/LitePT)).
Trained on DALES (University of Dayton) by the GeoScITY Lab, University of
Liege.

**SegFormer 3D.** The UrbanFiltering 3D SegFormer was developed by
**Zhouxin Xi** (tested by **Charumitha Selvaraj**) at the Canadian Forest
Service, Natural Resources Canada, as part of the
[TreeAIBox](https://github.com/NRCan/TreeAIBox) project. Crown Copyright,
Government of Canada.

**Dependencies.** The plugin builds on excellent open-source work:
[PyTorch](https://pytorch.org), [laspy](https://github.com/laspy/laspy),
[uv](https://github.com/astral-sh/uv),
[python-build-standalone](https://github.com/astral-sh/python-build-standalone),
[timm](https://github.com/huggingface/pytorch-image-models), and
[QGIS](https://qgis.org) itself.

**Author.** Abderrazzaq Kharroubi &mdash; GeoScITY Lab, University of Liege.
Bug reports and pull requests welcome on the
[issue tracker](https://github.com/akharroubi/AerialLidarClassifier/issues).


## Release setup and safety

Use **Plugins > Aerial LiDAR Classifier > Repair dependencies** to open setup.
Restart QGIS before replacing libraries already loaded by the classifier.
Setup accepts only prebuilt wheels and keeps TLS certificate verification
active throughout. Corporate certificates must be configured in the trust
store; certificate failures are not bypassed. Choose **Install CPU only** to
keep SegFormer off the GPU. LitePT always requires compatible NVIDIA CUDA
and spconv. A missing/incompatible spconv disables LitePT's Run button.

An environment is marked ready only after verification and a final cancellation
check. The new installation schema requests one rebuild of older environments.
Dependencies are stored separately, but inference still imports them into the
QGIS process: QGIS's already loaded NumPy and native libraries also matter.

All model files, including cached and imported weights, are checked against
release SHA-256 pins before deserialization. Custom URLs must serve the exact
released weights. Failed writes keep the previous output intact; outputs may
not alias any input in a batch. Truncated files must be repaired/re-exported
before classification. Geographic coordinates must be reprojected before use;
a units override is not a reprojection.

## Large data and resource use

Ordinary LAS/LAZ already use spatial tiling. Enable **streaming** when the full
point records do not fit in RAM; ordinary tiling alone still loads the full
input. Streaming stages tile data on disk and reads one buffered tile at a
time. It is not a fixed memory budget: dense tiles, buffers and model workspaces
still consume memory, and predictions/coverage still scale with total points.
Use a local SSD with sufficient temporary space and run one job at a time.

COPC indexing can accelerate spatial selection, but this release still uses
the verified sequential streaming path. A future indexed reader must query
full resolution, preserve original point identity/order, include buffer context
and write every point once. COPC level-of-detail queries would change the
inference inputs and cannot be advertised as equivalent full-resolution output.

Smaller tiles/crops/buffers may reduce peak memory but can change labels and
increase repeated work. LitePT retries GPU out-of-memory crops down to 5,000
points, then fails clearly if memory is still insufficient; retries can change
predictions. No GPU utilisation/energy cap or pause/resume is implemented.
To keep the GPU free, choose SegFormer and uncheck Use GPU (CPU work remains).
Disable automatic result loading during large batches to avoid simultaneous
QGIS rendering/indexing work. Clearing unused GPU cache does not free tensors
held by an active model.

Build the distributable with `python build_zip.py`. The ZIP contains one
`AerialLidarClassifier` folder, excludes environments/tests/weights/evidence,
and has a SHA-256 sidecar. Existing published tags are not rewritten by this tool.

Waveform packet payloads are explicitly refused because their relocation is
not implemented; export a point-only copy first.

## Learn with the plugin author

[LiDAR Point Clouds Processing in QGIS](https://maven.com/geomatics/qgis3d) is
Abderrazzaq Kharroubi's optional paid live cohort covering classification,
terrain models, COPC and 3D editing. View the syllabus and current dates on Maven.
The plugin panel, About dialog and plugin menu link to the course. The panel
invitation can be hidden. Links include only fixed campaign tags; no background
tracking or point-cloud data is sent.
