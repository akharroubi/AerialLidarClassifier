# Aerial LiDAR Classifier 1.1.0: final review before publication

Date: 25 September 2026. Base: Astra's remediation commit `4c48909`
(branch `codex/release-hardening`, built on `v1.1-dev` `e1dda5d`), kept
in full and reviewed file by file. This document lists what was changed
on top of it, why, and the evidence gathered. Astra's own report is
`RELEASE_READINESS_1.1.0.md`; the independent audit is
`audit_20260925/AUDIT_v1.1.0_ASTRA.md`.

## 1. Verdict

Release 1.1.0 today, as an experimental update of the existing listing.
Skip 1.0.3: it still carries the defects Astra reproduced (batch overwrite,
streaming metadata loss, TLS downgrade, source builds), and 1.1.0
supersedes it for every user.

## 2. Changes on top of Astra's candidate

| Area | Change | Reason |
|------|--------|--------|
| Packaging | The ZIP folder is `Aerial_LiDAR_Classifier`, not `AerialLidarClassifier`. New `build_zip.py` at the repository root: explicit file list, fixed timestamps, LF text, SHA-256 sidecar. | The published plugin is plugins.qgis.org/plugins/Aerial_LiDAR_Classifier. A ZIP with another folder name is another plugin for QGIS and for the repository: it would not update the listing, and users would end up with two copies. |
| QGIS 4 | `qgisMaximumVersion=4.99` after testing on QGIS 4.2.2 (section 3). All eight spots flagged by the official QGIS 4 checker now use scoped enums; `utils/compat.py` falls back to the legacy spelling only where a QGIS 3 release lacks the scoped one. | GitHub issue #6 and the owner's request. |
| Weights | Download automatically: Setup fetches the weights of every model the machine can run; the first run of a model (dock, Processing, `qgis_process`) fetches what is missing, with progress and cancel. Download dialog and button removed; "import a file" kept for offline machines. | Owner's directive: no extra step for users. |
| Dock | Two-row model strip (names no longer clipped); coloured status; one-click "Use SegFormer 3D" when the selected model cannot run; RTX 50 cards open on SegFormer 3D; a hint that says what is missing before Run; Run cannot start twice; elapsed time and an "Open folder" button when a run ends; secondary text readable on dark themes; progress bar shown only once a run starts; more room for the parameters. | UX request; real-GUI screenshots on QGIS 4 dark theme showed unreadable grey text and a squeezed panel. |
| Setup | States which models the machine gets; offers "Install the CPU version instead" after a failed GPU install (never silently); plain-language failure messages; the "torch already loaded" guard only blocks when the plugin's own torch is loaded. | Astra removed the silent CUDA to CPU fallback, which is right, but left users with a checkbox to find. |
| Course | Compact card under Run (hideable), full card in About, menu item, a course button in the completion message (hidden with the card), a line in the plugin manager's About text, and a callout near the top of the README. Links carry fixed UTM tags per placement (`panel`, `about`, `menu`, `success`, `listing`, `readme`). | Owner's request for conversion. The moment a run succeeds is the highest-intent placement. |
| QGIS 4 Setup | The installer's subprocesses no longer inherit `PYTHONEXECUTABLE` (set by QGIS 4's launcher), and Setup refuses to report ready unless the packages are inside the plugin's environment. | Found by a fresh first-run Setup on QGIS 4.2.2: the venv pointed at QGIS's `python3.exe` and uv installed all 4.6 GB into the portable Python. Setup said ready, then QGIS could not import torch and the Setup panel would have reopened at every start. QGIS 3.44 does not set that variable, which is why earlier installs worked. |
| Messages | A weights file that downloaded but could not be moved into place now says so (folder not writable, or a path over 260 characters) instead of blaming the network. | Found by the `qgis_process` test in a deep folder. |
| Output | The provenance VLR of a re-classified file replaces the previous one of the same field instead of stacking. | Re-running on an output added duplicate records. |
| Robustness | A Processing provider that fails to register no longer stops the dock from loading. | Astra's `initProcessing()` raised straight out of `initGui()`. |
| Docs | README rewritten against the code (install folder name, automatic weights, SegFormer geometry, defaults, Processing parameters, QGIS 4 paths, troubleshooting). Changelog and `metadata.txt` updated; About text names the external dependencies as the repository requires. | Several statements were wrong (Astra's S16). |
| Tests | Download dialog test replaced by tests of the automatic download, cancel and error messages, the torch guard, the provenance VLR and the enum helper. QGIS test harness imports QGIS before `unittest.mock`. | On QGIS 4 (OSGeo4W), importing `ssl` before `qgis.core` loads Python's own OpenSSL and QGIS's core library fails to load. This cannot happen inside QGIS or `qgis_process`. |

Kept from Astra without change: every data-integrity fix (atomic output,
raw extra bytes, EVLRs, truncation, protected fields, raw custom IDs,
batch aliasing), the installer policy (wheels only, TLS always verified,
no silent CPU fallback, staged Python extraction), weights verification
before every load, the 5,000-point OOM floor, units fixes, `initProcessing`,
licence notices.

## 3. Evidence

Machine: Windows 10 19045, RTX 3090 (driver 616.56).

| Check | QGIS 3.44.10 LTR (Qt 5.15) | QGIS 4.2.2 (Qt 6.11, PyQt 6.11, Python 3.12.14) |
|-------|----------------------------|-----------------------------------------------|
| QGIS release checks (17) | pass | pass |
| Unit and model tests (42 + 10) | pass (plugin Python) | same environment |
| Real GUI: QGIS loads the plugin from the ZIP, toolbar opens the dock, algorithm registered | pass | pass |
| Processing, LitePT-L CUDA, 2,708,283 pts | all points and XYZ kept, 54.8 s | identical labels to 3.44, 54.5 s |
| Processing, SegFormer 3D CUDA | 23.1 s | identical labels to 3.44, 23.2 s |
| Same tile in US survey feet (compound WKT) | units detected and converted | identical labels to 3.44 |
| Weights downloaded automatically by Processing | | both models, SHA-256 verified, byte-identical to the released files |
| `qgis_process plugins enable` + `run` (SegFormer 3D, CPU, weights downloaded on first use) | | pass, 433 k points in 54 s |
| Fresh first-run Setup through the plugin's worker thread | Astra: pass (478 s) | pass after the fix (366 s, CUDA torch 2.14 + spconv, both weights) |
| Classification with the environment that fresh Setup built | | both models, labels identical to 3.44 |

LitePT-L on QGIS 4 agrees with the reference implementation
(`mls_dales.infer`) on 99.9977 % of the 2,708,283 points (61 differ), the
same level as before. The feet and metre versions of the tile agree on
98.88 %: at 10 cm voxels, the rounding of converted coordinates moves some
points across voxel boundaries.

Packaging and repository scans on the final ZIP: official QGIS 4 checker
(`pyqt5_to_pyqt6.py`, run under QGIS 4) 0 findings; Bandit 0 HIGH and
0 MEDIUM (85 LOW, which do not block); detect-secrets 0 findings; ZIP
root `Aerial_LiDAR_Classifier`, 69 files, no tests, logs or weights,
identical bytes when rebuilt.

## 4. Fresh Setup on QGIS 4

Empty cache and empty profile, run through the plugin's own
`DepsInstallWorker` (QThread) under QGIS 4.2.2:

| Run | Result |
|-----|--------|
| Before the fix | Worker reported ready after 385 s, but the packages were in the portable Python and the venv was empty; the status check failed right after. |
| After the fix | Ready after 366 s: Python 3.12.8, uv 0.10.6, torch 2.14.0+cu126, torchvision 0.29.0+cu126, spconv-cu126 2.3.8, both weights downloaded and verified; `pyvenv.cfg` points at the portable Python; status "Ready". |
| Classification with that environment | LitePT-L 55.3 s and SegFormer 3D 23.8 s on 2,708,283 points; labels identical to the QGIS 3.44 runs. |

## 5. Not tested

- macOS and Apple MPS; Linux inside a QGIS GUI (the Linux installer was
  tested by Astra outside QGIS).
- QGIS 3.34 to 3.43: no install on this machine (the 3.40.5 folder is
  incomplete). The enum helper keeps the legacy spellings available.
- GPUs under 8 GB and RTX 50 cards on real hardware.
- A user whose profile path is long enough to exceed 260 characters with
  the weights file name; the error message now explains it.

## 6. After release

- Watch the plugin's "Qt6 Check" tab and the first issues from QGIS 4 users.
- Consider `experimental=False` once a few users confirm Setup on other
  machines: experimental plugins are hidden unless users tick *Show also
  experimental plugins*.
- Compare Maven sign-ups by `utm_content` to see which placement converts.
