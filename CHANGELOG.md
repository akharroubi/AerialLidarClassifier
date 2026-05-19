# Changelog

All notable changes to **Aerial LiDAR Classifier** will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project follows [Semantic Versioning](https://semver.org/).

## [1.0.3] - 2026-05-19

### Fixed
- **Linux first-install crashed at the venv pre-flight check** with
  `error while loading shared libraries: libpython3.12.so.1.0:
  cannot open shared object file` on every distro using
  python-build-standalone (i.e. all of them). Reported by
  @giswqs / qiusheng on Manjaro / QGIS 3.42 / RTX 6000 Ada.
  Reproduced cleanly in WSL Ubuntu by re-running the exact install
  steps.
  - **Root cause**: ``python -m venv --copies`` copies the
    python-build-standalone ``python3`` binary into ``venv/bin/`` but
    does NOT copy the ``libpython3.12.so.1.0`` next to it (venv has
    no notion that python-build-standalone ships libpython as a
    separate file rather than a static linker symbol). After the
    copy, the binary's ``RPATH=$ORIGIN/../lib`` resolves to the
    empty ``venv/lib/`` and every invocation dies at load time.
  - **Fix**: ``--copies`` was originally added to dodge Windows AV
    quarantine of the small "redirector launcher" the default
    symlink mode produces. It is now applied **only on Windows**.
    Linux and macOS use the symlink default, which points
    ``venv/bin/python3`` back at the standalone tree so the RPATH
    lookup still finds libpython. End-to-end verified in WSL
    Ubuntu: standalone Python downloads, symlink venv passes
    pre-flight, ``uv pip install numpy`` succeeds inside the venv.

## [1.0.2] - 2026-05-19

### Fixed
- **"Use GPU" checkbox stayed unchecked across QGIS sessions after a
  fresh install.** The dock's `closeEvent` persisted the checkbox state
  unconditionally, even when `_check_gpu()` had force-disabled and
  force-unchecked the box because the GPU probe failed (torch not yet
  importable on the very first launch, or a transient CUDA wheel /
  driver mismatch during the `cu128 → cu118` cascade). The `False`
  then stuck and left users silently on CPU on every subsequent
  launch, even after the CUDA install eventually succeeded. The
  plugin now only writes the GPU preference back to `QgsSettings`
  when the checkbox is actually enabled (i.e. the user had a real
  choice), and a one-time settings migration resets the stale `False`
  on first launch of >= 1.0.2 so existing v1.0.1 users get GPU
  acceleration back automatically.

- **Streaming I/O no longer silently drops the trailing partial chunk
  on very large LAS files.** v1.0.1 caught the laspy
  `ValueError: buffer size must be a multiple of element size`
  raised at the very last chunk (typical above ~2 GB / point format 6)
  and skipped that chunk to let the pipeline continue. The cost was
  that the streaming output had a few thousand fewer points than the
  input - a silent data loss. The chunk loop now uses
  `LasReader.read_points(n)` directly, so the alignment-mismatch path
  doesn't trigger on well-formed files. If it does trigger (genuine
  laspy edge case), a raw-byte recovery path reads the trailing bytes
  off `point_source.source`, trims to the largest whole-point-record
  slice, and decodes that slice via `PackedPointRecord.from_buffer`.
  The streaming output now has the same point count as the input.

### Changed
- **Trimmed the QGIS Plugin Manager *About* text** down to one short
  paragraph so the rating widget stays visible in the right panel
  without scrolling. Full feature list and installation details
  remain in `README.md` and the project homepage.

## [1.0.1] - 2026-05-15

### Fixed
- **SSL certificate errors when fetching from
  `download.pytorch.org`**. uv was using its bundled webpki roots,
  which do not include CAs that corporate IT installs via Group
  Policy for SSL inspection. The install pipeline now passes
  `--native-tls` so uv uses the OS certificate store (Schannel on
  Windows, Secure Transport on macOS, OpenSSL on Linux). The
  download.pytorch.org host is also added to the
  `--allow-insecure-host` list (uv) and `--trusted-host` list (pip
  fallback path), alongside the existing `pypi.org` and
  `files.pythonhosted.org` entries.
- **Linux first-install failed at `ensurepip`**. python-build-standalone
  Linux tarballs do not ship the bundled pip wheel under
  `Lib/ensurepip/_bundled/`, so the default `python -m venv` call
  exited 127 while bootstrapping pip and the whole install aborted.
  The venv is now created with `--without-pip` since we use `uv` for
  every package install regardless of platform.
- **CUDA wheel selection now cascades** through
  `cu128 → cu126 → cu124 → cu121 → cu118` instead of returning `None`
  the first time a candidate's driver requirement is unmet. NVIDIA
  drivers older than 550 (very common on Ampere-era workstations)
  recover GPU acceleration.
- **Torch version cap per CUDA index**. PyTorch's
  `download.pytorch.org/whl/cuXXX/` indexes also publish `+cpu`
  wheels of newer torch releases as a fallback. With
  `torch>=2.0.0,<3.0.0` and `--index-url cu121`, `uv` was picking the
  latest version (e.g. `torch 2.12.0+cpu`) instead of the latest
  cu121 wheel (e.g. `torch 2.5.1+cu121`). Caps applied: cu118 / cu121
  → `<2.6`, cu124 → `<2.8`.

### Added
- **Install marker** recording the plugin version that built the
  venv. On every dependency check the marker is compared against the
  running plugin version; mismatch (or missing marker) reopens the
  Setup dock with a one-click *Reinstall Dependencies* that wipes the
  stale cache before rebuilding. Users on the broken v1.0.0 venv no
  longer need to manually delete `~/.qgis_aerial_lidar_classifier/`
  when upgrading.
- **Reinstall now wipes first**: the dependency-install worker
  removes the entire cache directory before recreating the venv, so
  the recovery path is deterministic.
- **Auto 3D rendering** on loaded point-cloud layers. The plugin
  attaches a `QgsClassificationPointCloud3DSymbol` to the layer right
  after `addMapLayer`, so opening a 3D Map View renders the points in
  3D coloured by ASPRS class without any manual *Layer Properties*
  setup.

## [1.0.0] - 2026-05-19

### Added
- Initial public release on the QGIS plugin repository.

### Dependency management (rewritten)
- The plugin now installs PyTorch + LiDAR dependencies into an
  **isolated virtual environment** under
  `~/.qgis_aerial_lidar_classifier/venv_pyX.Y/` instead of touching the
  QGIS Python site-packages. This means no conflicts with QGIS itself
  or with other Python-heavy QGIS plugins.
- Setup pipeline: download a portable Python (matching the QGIS minor
  version) from python-build-standalone, download `uv`, create the
  venv, and install the wheels in one go.
- **NVIDIA GPU detection is now compute-capability- and driver-aware**.
  `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total`
  is parsed and the right CUDA wheel index is picked - including
  `cu128` for Blackwell / RTX 50-series and `cu126` on Windows when
  the driver supports it.
- A minimum-driver table per CUDA toolkit means we now refuse to
  install a wheel the user's driver cannot load, instead of failing
  later at `import torch`.
- Windows DLL search paths are registered automatically on every load.
- macOS Rosetta-on-Apple-Silicon detection warns about emulation.
- Pip / uv errors are pattern-matched (SSL, proxy, network, AV) to
  give the user actionable messages instead of stderr dumps.
- A dependency hash + install-logic version is persisted; bumping
  either forces a clean reinstall on next open. A "Reinstall" button
  in the setup dock lets the user trigger this manually.

### QGIS plugin guidelines compliance
- `metadata.txt` now carries an explicit `license=GPL-3.0-or-later`.
- The model downloader now uses `QgsBlockingNetworkRequest` so the
  user's QGIS proxy + authentication + certificate settings are
  respected (instead of calling `requests` directly).
- The `requests` package was dropped from REQUIRED_PACKAGES.

### Credits
- Semantic segmentation of aerial LiDAR point clouds into 7 ASPRS classes
  using the 3D SegFormer **UrbanFiltering** model from the
  [TreeAIBox](https://github.com/NRCan/TreeAIBox) project
  (Zhouxin Xi, tested by Charumitha Selvaraj — Natural Resources Canada,
  Crown Copyright, distributed under CC BY-NC 4.0).
- Native QGIS-style **dock panel** (right-side, collapsible groups,
  `QgsFileWidget`, `QgsCollapsibleGroupBox`, `QgsMessageBar`) replacing
  the previous floating dialog.
- QGIS **Processing algorithm** + **provider** so the classifier can be
  used from the Processing Toolbox, the Graphical Modeler and the
  `qgis_process` CLI.
- One-click dependency installer for PyTorch (CPU / CUDA 11.8 / CUDA 12.x),
  laspy, lazrs, timm, numpy_indexed and requests.
- Automatic NVIDIA GPU detection (CUDA) with CPU fallback and Apple
  Silicon MPS probing.
- On-demand model download to the QGIS profile cache with atomic file
  writes (no partial files left on disk on failure).
- Translation scaffolding via the standard `i18n/` directory.

### ASPRS compliance
- **Output is now fully ASPRS-compliant by default.** The classification
  is written to the standard LAS `classification` dimension (not an
  extra-byte field).
- The LAS file is automatically promoted to **point record format 6 and
  LAS 1.4** when the assigned ASPRS code exceeds the 5-bit limit of the
  legacy formats (so any code 0-255 can be encoded losslessly).
- A new **"Write to ASPRS-standard 'classification' field (recommended)"**
  toggle in the Output group lets advanced users opt out and write to a
  custom extra-byte field instead.

### Default class mapping
- The classifier always emits the standard ASPRS-1.4 codes:
  Ground (2), Vegetation (5), Building (6), Wires (14), Pole (15).
  Vehicles and Fences are mapped to **ASPRS 1 (Unclassified)** because
  the LAS spec has no dedicated code for them.
- The class mapping is **internal-only**: the dock no longer exposes a
  Class-mapping editor and the matching `class_manager_dialog.py` and
  QgsSettings persistence helpers were removed. The output is always
  ASPRS-compliant by construction, which is the only behaviour anyone
  actually needs in production.

### Output format (simplified)
- No output-format dropdown. The output extension mirrors the input:
  `.las` -> `.las`, anything else (including `.copc.laz`) -> `.laz`.
  COPC inputs are written as plain LAZ (the COPC spatial index is
  not regenerated). The `OUTPUT_FORMAT` parameter and the PDAL
  transcoder were removed.

### Spatial tiling for large files
- New **Performance / Tiling** group in the dock and matching
  `TILE_ENABLED` / `TILE_SIZE_M` / `TILE_BUFFER_M` parameters in the
  Processing algorithm.
- When enabled, the input is split into N x N spatial tiles with a
  configurable buffer halo. Each tile is classified independently,
  predictions for buffer points are discarded, and core predictions
  are merged back. Auto-size targets ~10 M points per tile.

### Preserve existing classes (removed)
- The "Preserve existing classes" workflow has been removed in
  favour of a simpler UX. The output is always overwritten with the
  model's predictions. Users who need partial preservation can run
  the classifier on a custom extra-byte field name and post-merge
  externally.

### Streaming I/O (files larger than RAM)
- New **Streaming I/O** sub-option under tiling (and `TILE_STREAMING`
  parameter in the Processing algorithm). Implements a 4-pass
  algorithm:
    1. Header scan to compute the tile grid.
    2. Stream-read the input via ``laspy.chunk_iterator`` and route
       points into per-tile disk-backed ``.npz`` sidecars.
    3. Per-tile inference loaded from disk; only core (non-buffer)
       predictions are written into a global predictions array
       indexed by the original point order.
    4. Stream-read the input again into a streaming writer that
       constructs each output chunk against the *output* schema,
       copies all common dimensions from the input by name, applies
       preserve-classes overrides, and sets the chosen classification
       field with the predictions slice.
- Memory footprint is roughly *(global predictions = 4 bytes / point) +
  (one tile in RAM) + (one chunk in the writer)*.
- Full feature parity with the in-memory path:
    - Writes to either the standard ASPRS ``classification`` dimension
      **or** a custom extra-byte field (the field is added to the
      output header before writing and populated chunk-by-chunk).
    - Applies the **Preserve existing classes** workflow per chunk
      against any chosen input dimension.
    - **Automatically upgrades the file to LAS 1.4 / point format 6**
      when an assigned ASPRS code exceeds the legacy 5-bit limit, by
      converting the header up-front and re-packing each chunk into
      the new point format on write.

### Fixes
- **COPC writing no longer crashes** with `'list' object has no
  attribute 'write_to'`. The COPC-VLR stripper now mutates the
  existing `laspy.vlrs.vlrlist.VLRList` in place instead of replacing
  it with a plain Python list, preserving the writer's expected API.
- **Auto-load failures are now visible.** When `QgsPointCloudLayer`
  reports an invalid output, the actual PDAL provider error surfaces
  in the panel's message bar (and the log), instead of disappearing
  into a silent warning.

### Result loading
- The **"Load classified files in QGIS after processing"** option in the
  dock now controls a dedicated row in the Output group.
- The same option is exposed as a `LOAD_AS_LAYER` parameter in the
  Processing algorithm so models and the `qgis_process` CLI can opt
  in/out as well.
