# Aerial LiDAR Classifier

Deep-learning semantic segmentation of aerial LiDAR point clouds (LAS / LAZ / COPC)
inside QGIS, using the 3D SegFormer architecture from the
[TreeAIBox](https://github.com/NRCan/TreeAIBox) project (Natural Resources Canada).

[![QGIS](https://img.shields.io/badge/QGIS-3.34%2B-1f9b4e)](https://qgis.org)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-3776AB)](https://www.python.org)
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
- [Tiling and streaming](#tiling-and-streaming)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Model weights](#model-weights)
- [Citation](#citation)
- [License](#license)
- [Credits](#credits)

---

## Features

- **Native dock-based UI** with drag-and-drop, layer picker, and live log.
- **Processing algorithm and provider** &mdash; usable from the Toolbox, the
  Graphical Modeler, and the `qgis_process` command-line interface.
- **Isolated dependency installer.** On first launch the plugin downloads a
  portable Python and the required wheels into a per-user virtual environment
  under `~/.qgis_aerial_lidar_classifier/`. The system Python and QGIS's own
  bundled Python are never touched. No admin rights required.
- **Automatic CUDA wheel selection.** NVIDIA driver and compute-capability are
  detected via `nvidia-smi`; the matching PyTorch wheel
  (`cu121` / `cu124` / `cu126` / `cu128`) is installed. CPU fallback on
  unsupported hardware. Apple Silicon uses MPS.
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
- PyTorch + supporting wheels (~1&ndash;3&nbsp;GB depending on CUDA),
- the SegFormer model weights (~18&nbsp;MB, SHA-256 verified).

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

The output preserves the input format, point format, scaling, CRS, and all
extra dimensions; only the `classification` dimension is rewritten.

---

## Usage

### Dock panel

The dock is organised as two stacked groups:

- **Input** &mdash; drop list, "Add files...", "Add folder...", layer picker.
- **Output** &mdash; folder, suffix (default `_classified`), format,
  classification field (default `classification`), and an "Auto-load result as
  layer" toggle.

Advanced parameters (collapsible) expose:

- **Compute device** &mdash; auto / CUDA / MPS / CPU.
- **Tiling** &mdash; enable, tile size (m), buffer (m), streaming I/O.

The bottom pane is always-visible log output with timestamps and severity
colouring; it mirrors the entries written to the QGIS Log Messages panel under
the *Aerial LiDAR Classifier* tag.

### Processing algorithm

The plugin registers a Processing provider, so the algorithm
`aerial_lidar:classify` is available from:

- the **Processing Toolbox**,
- the **Graphical Modeler** (chainable with any other Processing alg), and
- the `qgis_process` CLI:

```bash
qgis_process run aerial_lidar:classify \
  --INPUT=/data/tile.laz \
  --OUTPUT=/data/tile_classified.laz \
  --TILE_ENABLED=true \
  --TILE_SIZE_M=200 \
  --TILE_BUFFER_M=20
```

All UI options are exposed as algorithm parameters; advanced options
(`FIELD_NAME`, `TILE_STREAMING`, &hellip;) live under the Processing dialog's
*Advanced parameters* group.

---

## Hardware and platform support

| Backend            | OS                          | Tested |
|--------------------|-----------------------------|--------|
| NVIDIA CUDA 12.1   | Windows 10/11, Linux        | yes    |
| NVIDIA CUDA 12.4   | Windows 10/11, Linux        | yes    |
| NVIDIA CUDA 12.6   | Windows 10/11, Linux        | yes    |
| NVIDIA CUDA 12.8   | Windows 10/11, Linux        | yes    |
| Apple Silicon MPS  | macOS 13+                   | yes    |
| CPU only           | Windows / Linux / macOS     | yes    |

Recommended for production-sized tiles: an NVIDIA GPU with **&ge; 3&nbsp;GB**
VRAM. The default tile target of 10&nbsp;M points is sized for 3&nbsp;GB cards
at 30&nbsp;cm voxel resolution. CPU inference works but is roughly 50&ndash;100&times;
slower than a mid-range GPU.

The right CUDA wheel is selected automatically from `nvidia-smi`'s reported
compute capability *and* driver version; you do not need to install CUDA
yourself.

---

## Classification output

Output points use the ASPRS LAS 1.4 classification codes:

| Model class | Name        | ASPRS code | Meaning                  |
|------------:|-------------|-----------:|--------------------------|
| 1           | Ground      | 2          | Ground                   |
| 2           | Vegetation  | 5          | High Vegetation          |
| 3           | Vehicles    | 1          | Unclassified&nbsp;\*     |
| 4           | Wiring      | 14         | Wire &mdash; Conductor   |
| 5           | Fence       | 1          | Unclassified&nbsp;\*     |
| 6           | Pole        | 15         | Transmission Tower       |
| 7           | Building    | 6          | Building                 |

\* ASPRS LAS 1.4 has no dedicated codes for Vehicles or Fences, so these are
mapped to **1 = Unclassified** by default. If you need them as a separate
class, override `FIELD_NAME` in the Processing algorithm to write to an extra
LAS dimension instead.

When the input file already uses Point Record Format &lt; 6, the writer
upgrades it to PRF 6 / LAS 1.4 (required for ASPRS codes &gt; 31). All other
attributes (RGB, intensity, return number, GPS time, scan angle, extra
dimensions) are preserved byte-for-byte.

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

The model is the **UrbanFiltering 3D SegFormer**
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
into `~/.qgis_aerial_lidar_classifier/models/` &mdash; the plugin will
verify its SHA-256 and use it without ever touching the network.

### Install takes a long time / progress bar appears stuck

The PyTorch + CUDA wheels are 1&ndash;3&nbsp;GB depending on which CUDA
version was selected. On a slow connection the download can genuinely take
ten minutes or more. The Log Messages panel shows the live progress; if
you see new lines, the install is still working.

---

## Model weights

The bundled model is downloaded on first use:

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

If you use this plugin in academic work, please cite the underlying model:

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
  version      = {1.0.0},
  note         = {GPL-3.0-or-later}
}
```

---

## License

- **Plugin source code** &mdash; [GPL-3.0-or-later](LICENSE).
- **Model weights** &mdash; [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
  &copy; the model authors. Distributed unchanged from the upstream
  TreeAIBox release. Commercial users must obtain a separate licence from
  the model author.

---

## Credits

**Model.** The UrbanFiltering 3D SegFormer was developed by
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
