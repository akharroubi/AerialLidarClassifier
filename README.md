# LiDAR AI Classifier

<p align="center"><img src="assets/logo_128.png" alt="LiDAR AI Classifier logo" width="128" height="128"></p>

Deep-learning semantic segmentation of airborne and mobile mapping LiDAR
(LAS / LAZ / COPC) inside QGIS 3.34+ and QGIS 4, with three models:

- **LitePT-L Airborne** (default on NVIDIA GPUs): a point transformer from
  [prs-eth/LitePT](https://github.com/prs-eth/LitePT) using 10 cm voxels and
  eight classes.
- **LitePT-L Mobile Mapping** (new in 1.2): a separate model using the same
  architecture with 5 cm voxels. Nine classes (ground, low vegetation,
  high vegetation, building, pole-like, vehicle, fence/barrier, wire, unknown),
  XYZ only, with editable output codes. NVIDIA GPU required.
- **SegFormer 3D Airborne** (UrbanFiltering, [TreeAIBox](https://github.com/NRCan/TreeAIBox),
  Natural Resources Canada): a voxel transformer at 30 cm that also runs on the
  CPU and on Apple Silicon.

Until version 1.1 the plugin was called **Aerial LiDAR Classifier**. The
installation folder is still `Aerial_LiDAR_Classifier`, so an update keeps
saved settings, downloaded weights and Processing models.

[![QGIS](https://img.shields.io/badge/QGIS-3.34%20to%204.x-1f9b4e)](https://qgis.org)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-3776AB)](https://www.python.org)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![Model: CC BY-NC 4.0](https://img.shields.io/badge/model-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-555)](#hardware-and-platform-support)

Every input point is written back with a LAS classification code. The airborne
models use the [ASPRS LAS 1.4](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)
codes; the mobile mapping model uses ASPRS codes where they exist and the
user-definable codes 64, 65 and 66 for poles, vehicles and fences, all editable.
The classified output is an ordinary LAS or LAZ file. For QGIS builds that
reject files with existing extra attributes, the plugin can prepare a COPC
viewing copy when loading the result.

> **Learn it with the author.** Abderrazzaq Kharroubi, who wrote this plugin,
> teaches *LiDAR Point Clouds Processing in QGIS*, a live cohort on
> classification, terrain models, COPC and 3D editing.
> [See the syllabus and the next dates](https://maven.com/geomatics/qgis3d?utm_source=github&utm_medium=readme&utm_campaign=qgis3d).
> The course is optional and paid; the plugin is and stays free.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Hardware and platform support](#hardware-and-platform-support)
- [Classification output](#classification-output)
- [Coordinate units](#coordinate-units)
- [Tiling, streaming and large files](#tiling-streaming-and-large-files)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Model weights](#model-weights)
- [Citation](#citation)
- [License](#license)
- [Credits](#credits)
- [Development](#development)

---

## Features

- **Three models, one interface.** Pick LitePT-L Airborne, LitePT-L Mobile
  Mapping or SegFormer 3D Airborne in the dock or with the `MODEL` parameter in
  Processing. If the selected model cannot run on the computer, the dock says
  why; for an airborne model it offers the other one in one click.
- **Your class codes for mobile mapping.** Each of the nine mobile mapping
  classes gets the code you choose (0 to 255), saved per model, or merged with
  another class by giving both the same code.
- **Nothing to install by hand.** On first use a Setup panel downloads a
  portable Python, PyTorch and the LiDAR libraries into a per-user folder
  (no admin rights), then the model weights. QGIS's own Python is left alone.
- **Weights download by themselves.** Setup fetches them; a model used for the
  first time (dock, Processing or `qgis_process`) fetches whatever is missing.
  Every file is checked against its published SHA-256 before each use.
- **Right PyTorch build for the GPU.** The NVIDIA driver and compute
  capability are read with `nvidia-smi` and the matching CUDA build of PyTorch
  is installed (`cu118` to `cu128`, `cu126` preferred), plus `spconv` for
  LitePT-L. Without an NVIDIA GPU the CPU build is installed. Apple Silicon
  uses MPS (SegFormer 3D).
- **Units handled.** Files in feet are converted from their CRS before
  inference; see [Coordinate units](#coordinate-units).
- **Large files.** Spatial tiling with a context buffer, and a streaming mode
  for files larger than RAM.
- **Processing algorithm** usable from the Toolbox, the Graphical Modeler and
  the `qgis_process` command line.
- **Safe output.** Results are written to a temporary file and published only
  when complete; a batch can never overwrite one of its own inputs.
- **Background execution** with progress, cancel and optional loading of the
  results as point-cloud layers.

---

## Installation

### From the QGIS plugin repository (recommended)

1. *Plugins > Manage and Install Plugins...*
2. Search for **LiDAR Classifier** (the plugin was called *Aerial LiDAR
   Classifier* before version 1.2).
3. Click **Install Plugin**.

### From source

```bash
git clone https://github.com/akharroubi/AerialLidarClassifier.git Aerial_LiDAR_Classifier
```

The folder must be named `Aerial_LiDAR_Classifier` (the plugin's package name).
Copy it into your QGIS plugins directory, restart QGIS and enable it in the
plugin manager:

| OS      | QGIS 3 plugins directory |
|---------|--------------------------|
| Windows | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` |
| macOS   | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |
| Linux   | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |

QGIS 4 uses the same paths with `QGIS4` in place of `QGIS3`.

### First launch

The first time you open the plugin, a **Setup** panel shows what it will do
and which models the computer will get. One click on *Install Dependencies*
downloads:

- a standalone Python interpreter (~30 MB) and
  [`uv`](https://github.com/astral-sh/uv), a fast package installer,
- PyTorch and the LiDAR packages (~1 to 3 GB depending on CUDA), plus `spconv`
  (~100 MB) on NVIDIA GPUs,
- the model weights: LitePT-L Airborne and LitePT-L Mobile Mapping (~172 MB
  each) on NVIDIA GPUs, SegFormer 3D (~18 MB) everywhere.

It usually takes 5 to 15 minutes, and QGIS stays usable meanwhile. Everything
is stored under `~/.qgis_aerial_lidar_classifier/` (the weights under the
QGIS profile) and can be removed at any time.

If the GPU installation fails, the Setup panel offers **Install the CPU version
instead** (SegFormer 3D on the CPU). The plugin never switches from GPU to CPU
by itself.

---

## Quick start

1. Click the plugin's toolbar icon (or *Plugins > LiDAR AI Classifier*).
2. Drag `.las` / `.laz` / `.copc.laz` files into the input list, or pick point
   cloud layers already loaded in the project.
3. Choose the model for the acquisition: **LitePT-L Mobile Mapping** for
   mobile mapping (vehicle, backpack) data, an **Airborne** model for aerial
   LiDAR. For mobile mapping, *MMS classes: Edit output codes...* sets the
   code of each class.
4. Choose an output folder (it defaults to the folder of the first file).
5. Click **Run classification**.

The line above the progress bar always says what is still missing before Run,
or what Run will do. When a run ends, the QGIS message bar shows the elapsed
time and an *Open folder* button.

---

## Usage

### Dock panel

- **Model strip** (top): the model selector, its status (ready, weights
  download on first run, needs an NVIDIA GPU, LitePT dependencies missing), the
  compute device, and a folder icon to import a weights file on offline
  machines.
- **Input files**: drop list, *Add files*, *Add folder*, and the loaded-layer
  pickers.
- **Output**: folder, filename suffix (default `_classified`), label field
  (default `classification`; airborne models only), *MMS classes: Edit output
  codes...* (mobile mapping model only), and *Load classified files in QGIS*.
- **Advanced parameters** (collapsed): *Use GPU*, *Clear GPU memory*, tiling
  (auto size or tile size in metres, buffer in metres), *Streaming I/O*, and
  *Input units*.
- **Log** (always visible) and the footer with progress, *Cancel* and *Run*.

### Processing algorithm

The algorithm `aeriallidar:classify_lidar` is available from the
**Processing Toolbox**, the **Graphical Modeler** (the classified file is the
`OUTPUT_FILE` output, ready to chain) and the `qgis_process` command line:

```bash
qgis_process run aeriallidar:classify_lidar -- \
  INPUT=/data/tile.laz OUTPUT_FOLDER=/data/classified \
  MODEL=0 TILE_ENABLED=true TILE_SIZE_M=0 TILE_BUFFER_M=50
```

| Parameter | Values |
|-----------|--------|
| `INPUT` | LAS, LAZ or COPC file |
| `MODEL` | `0` LitePT-L Airborne, `1` SegFormer 3D Airborne, `2` LitePT-L Mobile Mapping |
| `OUTPUT_FOLDER`, `SUFFIX` | output location, default suffix `_classified` |
| `DEVICE` | `0` Auto, `1` GPU (CUDA), `2` CPU, `3` GPU (Apple MPS) |
| `FIELD_NAME` | `classification` (default). Airborne models also accept a new name (raw model classes in an extra field) |
| `OUTPUT_CODES_JSON` | mobile mapping only, optional: codes keyed by model class ID, e.g. `{"5":20,"6":21,"7":22}`; blank keeps the defaults |
| `LOAD_AS_LAYER` | add the result to the project |
| `TILE_ENABLED`, `TILE_SIZE_M`, `TILE_BUFFER_M` | tiling; size `0` is automatic (about 10 M points per tile), buffer default 50 m |
| `TILE_STREAMING` | streaming I/O for files larger than RAM |
| `UNITS` | `0` auto-detect, `1` metres, `2` international feet, `3` US survey feet |

For example, mobile mapping with your own codes for poles, vehicles and fences:

```bash
qgis_process run aeriallidar:classify_lidar -- \
  INPUT=/data/street.laz OUTPUT_FOLDER=/data/classified MODEL=2 \
  'OUTPUT_CODES_JSON={"5":20,"6":21,"7":22}'
```

The weights download automatically on the first run of a model. The
dependencies themselves are installed once from the dock's Setup panel.

---

## Hardware and platform support

| Environment | Status for 1.2.0 |
|-------------|------------------|
| Windows 10, QGIS 3.44.10 LTR (Qt 5.15), Python 3.12, RTX 3090 | Tested: the three models on CUDA (SegFormer also on CPU), dock, Processing, `qgis_process`, data-integrity suite |
| Windows 10, QGIS 4.2.2 (Qt 6.11, PyQt 6.11), Python 3.12, RTX 3090 | Tested: plugin load, dock, Processing, `qgis_process`, the three models on CUDA |
| Linux x86_64 (Ubuntu under WSL), Python 3.10 | Installer and runtime tested outside the QGIS GUI |
| macOS, Apple Silicon (MPS) | Code path present, not tested; reports welcome |
| GPUs under 8 GB, other GPU generations | Not tested on real hardware; a smaller-crop path and out-of-memory retries exist |

**LitePT-L needs an NVIDIA GPU and compatible PyTorch and `spconv` packages.**
This installer selects `spconv` packages for its `cu118`, `cu121`, `cu124`
and `cu126` setup paths. Its Blackwell setup path uses `cu128` and does not
install `spconv`. The dock checks the installed packages and reports why a
selected model is unavailable. Check that message before choosing a model;
SegFormer 3D is also available for airborne data.

LitePT-L Airborne processes crops of up to 70,000 points within 30 m; cards
under 8 GB use 35,000-point / 20 m crops. LitePT-L Mobile Mapping processes
crops of up to 240,000 points within 12 m; cards under 16 GB use 120,000
points / 9 m and cards with 8 GB or less 50,000 points / 6 m. A crop that runs
out of memory is retried with half as many points, down to 5,000. Smaller crops give
the model less context and can change a few labels; the log says when a
reduced budget is used. Memory use and processing speed depend on the input
and hardware.

**SegFormer 3D** runs on the CPU, on NVIDIA GPUs and (untested) on Apple MPS.
On the CPU it is much slower than on a GPU but works on any computer.

You never need to install CUDA yourself: the PyTorch wheels carry it.

---

## Classification output

With the default `classification` field, points get the codes below.

**LitePT-L Mobile Mapping** (editable codes):

| Model class | Name            | Default code | Meaning                         |
|------------:|-----------------|-------------:|---------------------------------|
| 1           | Ground          | 2            | Ground                          |
| 2           | Low vegetation  | 3            | Low Vegetation                  |
| 3           | High vegetation | 5            | High Vegetation                 |
| 4           | Building        | 6            | Building                        |
| 5           | Pole like       | 64           | user-defined (poles, lamp posts, signs, traffic lights) |
| 6           | Vehicle         | 65           | user-defined (cars, vans, trucks, buses) |
| 7           | Fence barrier   | 66           | user-defined (fences, railings, guardrails, barriers) |
| 8           | Wire            | 14           | Wire, Conductor                 |
| 9           | Unknown         | 1            | Unclassified (street furniture, pedestrians, bicycles...) |

ASPRS LAS 1.4 has no code for poles, vehicles or fences, so they get codes
from the user-definable range 64 to 255 by default. To change any code, choose
the mobile mapping model and click *MMS classes: Edit output codes...*: the
table takes values from 0 to 255, is saved for the model, and *Reset to model
defaults* restores it. Giving two classes the same code merges them (for
example vehicles and unknown both to 1). In Processing, pass the codes as
`OUTPUT_CODES_JSON` keyed by model class, for example `{"5":20,"6":21,"7":22}`;
omitted classes keep their default, and the dock's saved table is not used.

The mobile mapping model always writes to the `classification` field. Codes
above 31 need LAS 1.4; legacy files are upgraded automatically (see below).

**LitePT-L Airborne**:

| Model class | Name        | ASPRS code | Meaning              |
|------------:|-------------|-----------:|----------------------|
| 1           | Ground      | 2          | Ground               |
| 2           | Vegetation  | 5          | High Vegetation      |
| 3           | Cars        | 1          | Unclassified \*      |
| 4           | Trucks      | 1          | Unclassified \*      |
| 5           | Power lines | 14         | Wire, Conductor      |
| 6           | Fences      | 1          | Unclassified \*      |
| 7           | Poles       | 15         | Transmission Tower   |
| 8           | Buildings   | 6          | Building             |

**SegFormer 3D** (UrbanFiltering classes):

| Model class | Name        | ASPRS code | Meaning              |
|------------:|-------------|-----------:|----------------------|
| 1           | Ground      | 2          | Ground               |
| 2           | Vegetation  | 5          | High Vegetation      |
| 3           | Vehicles    | 1          | Unclassified \*      |
| 4           | Wiring      | 14         | Wire, Conductor      |
| 5           | Fence       | 1          | Unclassified \*      |
| 6           | Pole        | 15         | Transmission Tower   |
| 7           | Building    | 6          | Building             |

\* ASPRS LAS 1.4 has no codes for vehicles or fences. To keep them apart with
an airborne model, set the label field to a new name (for example
`ai_label`): the model's own class numbers above are then written to that
extra field, and the input's `classification` is kept unchanged. Standard LAS
dimension names (`X`, `intensity`, `red`...) are refused as label fields.

What the output keeps:

- the same number of points, in the same order, with the same coordinates,
  scales, offsets, CRS, VLRs and EVLRs, and every other attribute byte for
  byte (including scaled 64-bit extra bytes);
- the input's format: LAS in gives LAS out, LAZ or COPC in gives LAZ out (a
  COPC input becomes ordinary LAZ, since its spatial index would no longer
  match);
- the point format, unless a code above 31 needs LAS 1.4 (formats 0/1 become
  6, 2/3 become 7, 4 becomes 9, 5 becomes 10; RGB is kept). The legacy scan
  angle becomes the LAS 1.4 scan angle (rounded to at most 0.003 degrees); the
  mobile mapping model adds no extra dimension, the airborne models also keep
  the original value in a `scan_angle_rank` extra field.

A small VLR (`AerialLiDAR`, record 1) records the model, its weights hash, the
field and the class mapping. Files that carry waveform packets are refused
(their relocation is not supported): export a point-only copy first.
Truncated files are refused instead of producing a shorter output.

Some QGIS builds refuse to open valid LAS 1.4 files that carry extra
dimensions (seen with the PDAL reader of QGIS 3.44). When *Load classified
files in QGIS* is ticked and the output keeps extra dimensions from the input,
the plugin also writes a viewing copy, `<output>.<id>.qgis-view.copc.laz`,
with QGIS's own Untwine, checks its point count, bounds and class counts, and
loads that copy instead. The classified LAS/LAZ (and the Processing
`OUTPUT_FILE`) is the result to keep: the viewing copy reorders points for its
spatial index and can be deleted at any time. If Untwine is missing or fails,
the log says why and the classified file is unaffected.

---

## Coordinate units

The models work in **metres**. Most US LiDAR is delivered in US survey feet;
fed to a model unconverted, every distance is 3.28 times too small, buildings
come out as *Wire, Conductor* and *Transmission Tower*, and the run is about 13
times slower.

The plugin reads the linear unit from the file's CRS (the WKT record of LAS 1.4
files, or the GeoTIFF keys of older files, including a vertical CRS code),
converts XY and Z to metres **for the model only**, and logs what it found:

```
tile.las: coordinate units: XY in US survey foot, Z in US survey foot
(WKT CRS: US survey foot horizontal, US survey foot vertical); converting to metres for the model
```

The output keeps the original coordinates, scales and CRS.

- Without a CRS in the header, metres are assumed and the log says so.
- When the CRS has no vertical part, Z is assumed to share the horizontal unit.
- Geographic files (degrees) are refused, even with a units override: reproject
  them first, for example to the local UTM zone.

To force a unit when the header is missing or wrong, use *Advanced parameters >
Input units* in the dock (auto-detect, metres, international feet, US survey
feet) or the `UNITS` parameter in Processing.

---

## Tiling, streaming and large files

- **Tiling** splits the file into square tiles with a context buffer; only the
  core of each tile is written. Tile size is automatic (about 10 M points per
  tile) or set in metres; the buffer defaults to 50 m. Tiling bounds the work
  handed to the model, but the whole file is still read into RAM.
- **Streaming I/O** (needs tiling; the dock ticks both) reads the file in
  chunks, keeps per-tile points in temporary files on disk, classifies one tile
  at a time and writes the output as it goes. Use it when the file does not fit
  in RAM, with enough free space on a local disk for the temporary files.

Streaming is not a fixed memory budget: a dense tile, its buffer and the model
still need memory, and the predictions take about 2 bytes per point. Run one
job at a time on very large data, and untick *Load classified files in QGIS*
for long batches.

To keep the GPU free for other work, choose SegFormer 3D and untick *Use GPU*
(the CPU then does the work). *Clear GPU memory* releases unused cached memory
between runs.

---

## How it works

**LitePT-L Airborne** works on points. Per tile the plugin removes an integer origin,
keeps one representative per occupied 10 cm voxel (with an exact map back to
every raw point), and covers the representatives with crops of at most 70,000
points within 30 m. Each crop is centred, floor-referenced and voxelised;
the network's class probabilities are averaged over every
visit of a point, and the best class is projected back to the raw points. The
upstream code (MIT) is vendored in `core/litept/` with pure-PyTorch replacements
for FlashAttention, `torch_scatter` and the RoPE kernel, so only `spconv` is a
compiled dependency.

**SegFormer 3D** is the UrbanFiltering 3D SegFormer
([Xi et al., TreeAIBox](https://github.com/NRCan/TreeAIBox)). Points are
voxelised on a 0.3 x 0.3 x 0.2 m grid (occupancy only), the grid is cut into
overlapping blocks of 112 x 112 x 256 voxels, the network labels every occupied
voxel (7 classes), and each point takes the label of its voxel.

**LitePT-L Mobile Mapping** uses the same architecture and the same steps
with 5 cm voxels and crops of at most 240,000 points within 12 m (centres
11 m apart), one pass. Its features are geometric only (height above the
crop), so intensity, colour and returns are not needed.

All models run inside a QGIS background task, so QGIS stays responsive.

---

## Troubleshooting

Every install and detection step is logged in *View > Panels > Log Messages*
under the **LiDAR AI Classifier** tag: start there.

### Repair or reinstall the dependencies

*Plugins > LiDAR AI Classifier > Repair dependencies* opens the Setup panel.
If the classifier was already used in this QGIS session, restart QGIS first
(Windows keeps the loaded libraries locked), then run Repair before opening the
classifier.

### The GPU installation failed

Update the NVIDIA driver from
[nvidia.com](https://www.nvidia.com/Download/index.aspx) (Game Ready or Studio
driver, not a partial driver from Windows Update) and click *Reinstall
Dependencies*, or click *Install the CPU version instead*.

### LitePT-L says "Needs an NVIDIA GPU", "GPU use is switched off" or "LitePT dependencies missing"

The LitePT-L models run only on NVIDIA GPUs, with the `spconv` library of the
plugin's environment.

- *GPU use is switched off*: tick *Use GPU* in *Advanced parameters*.
- *PyTorch is CPU-only*: run *Plugins > LiDAR AI Classifier > Repair
  dependencies*, leave the CPU option unticked, and restart QGIS.
- *LitePT dependencies missing*: the status tooltip and the dock log name the
  library that failed to load and the PyTorch/CUDA build in use. Restart QGIS,
  run *Repair dependencies*, restart again. If it persists, open an issue with
  that log text.
- RTX 50 cards (Blackwell): the installer's `cu128` path does not install
  `spconv`. Read the LitePT status and setup log for the packages available
  in your environment. SegFormer 3D is available for airborne data.

### Install fails with `WinError 4551` / "Application Control policy"

This is Windows Application Control (AppLocker, WDAC or Smart App Control), an
allow-list enforced by IT. It refuses to run any program that is not on the
list, and neither antivirus exclusions nor `AERIAL_LIDAR_CLASSIFIER_CACHE_DIR`
help. Ask IT:

> Please allow execution under `C:\Users\<my-username>\.qgis_aerial_lidar_classifier\`
> for my account, by adding the folder to the AppLocker / WDAC allow list or by
> signing the python-build-standalone binaries used by this QGIS plugin.

### Install fails with "blocked by antivirus / endpoint-security product"

Windows Defender or another antivirus quarantined the freshly downloaded
`python.exe`. With admin rights, from an elevated PowerShell:

```powershell
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.qgis_aerial_lidar_classifier"
```

Then click *Reinstall Dependencies*. Without admin rights, ask IT for that
exclusion, or move the install to a folder your organisation already allows:

```powershell
[Environment]::SetEnvironmentVariable("AERIAL_LIDAR_CLASSIFIER_CACHE_DIR", "C:\Dev\aerial_lidar_classifier", "User")
```

and restart QGIS.

### Certificate errors behind a corporate proxy

Certificate checks stay on; the installer never downloads packages over an
unverified connection. Install your organisation's root certificate in the
Windows (or system) certificate store, set the proxy in *Settings > Options >
Network*, and retry. SOCKS proxies are not supported by the installer.

### The weights do not download

They download through the QGIS network stack, which uses the proxy and
certificates set in *Settings > Options > Network*. On a machine without
internet access, download the `.pth` file from [Model weights](#model-weights)
elsewhere, then use the folder icon next to the model selector to import it;
the plugin copies it into
`<QGIS profile>/AerialLidarClassifier/models/<model id>/` and checks its
SHA-256.

### Almost everything is classified as wires or towers, or the run is very slow

The file is in feet and its CRS is missing or wrong. Set *Input units* (dock)
or `UNITS` (Processing). See [Coordinate units](#coordinate-units).

### The run uses the CPU although the computer has an NVIDIA GPU

Open *Advanced parameters* and tick *Use GPU*. If the box is greyed out, the
CPU version of PyTorch is installed: run *Repair dependencies* and leave
*Install the CPU version only* unticked.

### `Could not open file as point cloud layer` when loading the result

Your QGIS build lacks the PDAL provider. Open the file with *Layer > Add Layer >
Add Point Cloud Layer*; if that also fails, use a QGIS package that includes
PDAL (the official Windows and macOS installers do).

---

## Model weights

The weights download automatically. The URLs below are for offline machines.

**LitePT-L Mobile Mapping (MLS, 5 cm)**

| Field         | Value |
|---------------|-------|
| File          | `litept_l_mls_5cm_fp16.pth` (stored as float16, run in float32) |
| Size          | ~172 MB |
| SHA-256       | starts with `d2fd555c9147`; the full value is in `core/registry.py` and the plugin checks it automatically |
| URL           | [release v1.2 of this repository](https://github.com/akharroubi/AerialLidarClassifier/releases/tag/v1.2) |
| License       | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

**LitePT-L Airborne (10 cm)**

| Field         | Value |
|---------------|-------|
| File          | `litept_l_dales_10cm_ema_fp16.pth` (stored as float16, run in float32) |
| Size          | ~172 MB |
| SHA-256       | starts with `849ba5089e62`; the full value is in `core/registry.py` and the plugin checks it automatically |
| URL           | [release v1.1 of this repository](https://github.com/akharroubi/AerialLidarClassifier/releases/tag/v1.1) |
| License       | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

**SegFormer 3D Airborne (UrbanFiltering, 30 cm)**

| Field         | Value |
|---------------|-------|
| File          | `urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth` |
| Size          | ~18 MB |
| SHA-256       | starts with `cddb791041d4`; the full value is in `core/registry.py` and the plugin checks it automatically |
| Primary URL   | [NRCan/TreeAIBox release v1.0](https://github.com/NRCan/TreeAIBox/releases/tag/v1.0) |
| Mirror URL    | [release v1.0.0 of this repository](https://github.com/akharroubi/AerialLidarClassifier/releases/tag/v1.0.0) |
| License       | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

**Commercial use of the model weights requires a separate licence from the
model authors.** The plugin source code is GPL-3.0-or-later.

---

## Citation

If you use this plugin in academic work, please cite the model you used. For
LitePT-L, cite the LitePT architecture (see the
[LitePT repository](https://github.com/prs-eth/LitePT) for the paper).

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
  title        = {{LiDAR AI Classifier}: a QGIS plugin for deep-learning
                  semantic segmentation of airborne and mobile mapping
                  LiDAR point clouds},
  year         = {2026},
  url          = {https://github.com/akharroubi/AerialLidarClassifier},
  version      = {1.2.0},
  note         = {GPL-3.0-or-later}
}
```

---

## License

- **Plugin source code**: [GPL-3.0-or-later](LICENSE). The vendored LitePT
  model code (`core/litept/`) is MIT, (c) Photogrammetry and Remote Sensing Lab,
  ETH Zurich (see `core/litept/LICENSE.upstream`).
- **Model weights**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  The LitePT-L weights are provided by the plugin maintainer. The SegFormer 3D
  weights are distributed unchanged from TreeAIBox, whose authors permitted
  their use in this plugin. See
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and scope.

---

## Credits

**LitePT-L.** Architecture and reference implementation by the Photogrammetry
and Remote Sensing Lab, ETH Zurich ([prs-eth/LitePT](https://github.com/prs-eth/LitePT)).
Plugin integration and model weights provided by Abderrazzaq Kharroubi,
GeoScITY Lab, University of Liege.

**SegFormer 3D.** The UrbanFiltering 3D SegFormer was developed by
**Zhouxin Xi** (tested by **Charumitha Selvaraj**) at the Canadian Forest
Service, Natural Resources Canada, as part of the
[TreeAIBox](https://github.com/NRCan/TreeAIBox) project. Crown Copyright,
Government of Canada.

**Dependencies.** [PyTorch](https://pytorch.org), [laspy](https://github.com/laspy/laspy),
[spconv](https://github.com/traveller59/spconv), [uv](https://github.com/astral-sh/uv),
[python-build-standalone](https://github.com/astral-sh/python-build-standalone),
[timm](https://github.com/huggingface/pytorch-image-models), and
[QGIS](https://qgis.org) itself.

**Author.** Abderrazzaq Kharroubi, GeoScITY Lab, University of Liege. Bug
reports and pull requests are welcome on the
[issue tracker](https://github.com/akharroubi/AerialLidarClassifier/issues).

**Course.** [LiDAR Point Clouds Processing in QGIS](https://maven.com/geomatics/qgis3d?utm_source=github&utm_medium=readme&utm_campaign=qgis3d),
a live cohort by the author. The plugin links to it from the dock (the card can
be hidden with its x button), the About dialog and the plugin menu. The links
carry fixed campaign tags only; the plugin sends no usage data.

---

## Development

- `python build_zip.py` builds `dist/lidar_ai_classifier_v<version>.zip`
  with the `Aerial_LiDAR_Classifier` folder (the published package name), a
  fixed file list and a SHA-256 sidecar. The same source gives the same bytes.
  The build stops if the ZIP would exceed 25 MB, contain weights or binaries,
  or fail Bandit, Flake8's blocking checks or the secrets scan.
- Tests live in `tests/` (not shipped). `python tests/run_release_checks.py`
  runs everything and prints one table: the pytest suite with the plugin's
  Python (`~/.qgis_aerial_lidar_classifier/venv_py3.12`), then each
  `tests/qgis_*.py` script under QGIS 3 and QGIS 4 (`python-qgis-ltr.bat`,
  `python-qgis.bat`; set `ALC_QGIS3_PYTHON` / `ALC_QGIS4_PYTHON` to your
  launchers). `--skip-gpu` leaves out the real CUDA run.
