"""Real QGIS/CUDA end-to-end check of the Mobile Mapping model.

Run with QGIS's Python (python-qgis-ltr.bat on QGIS 3, python-qgis.bat on
QGIS 4) on a machine with an NVIDIA GPU and the plugin's environment.

Uses the 80,031-point reference sample of the maintainer's
validation folder (not redistributed) and the released weights: the file
is imported through the plugin's own import path (SHA-256 checked) into
the standalone QGIS profile, never into the user's QGIS profile.

Checks, for in-memory and streaming modes with custom output codes:
same point count and order, every source attribute unchanged, the input
file unchanged, LAS 1.4 upgrade for codes above 31, the mapping VLR, a
COPC viewing copy that QGIS opens with the same point count, identical
labels between both modes and direct backend inference on the same LAS
coordinates, and agreement with the reference predictions.

Writing the sample to LAS rounds coordinates by up to 5 micrometres, which
alone changes about 0.75 % of the labels of the float64 sample (measured
with the float32 weights too), so the reference agreement bound is 99 %;
the exact comparison is the direct inference on the LAS coordinates.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from qgis.core import QgsApplication, QgsPointCloudLayer
app = QgsApplication([], False)
app.initQgis()

import hashlib
import importlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent / "validation"
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(Path.home() / ".qgis_aerial_lidar_classifier/venv_py3.12/Lib/site-packages"))
import laspy
import numpy as np

tasks = importlib.import_module(ROOT.name + ".workers.classifier_task")
registry = importlib.import_module(ROOT.name + ".core.registry")
manager_module = importlib.import_module(ROOT.name + ".utils.model_manager")
MLS = registry.LITEPT_L_MLS
torch = tasks._get_torch()
if not torch.cuda.is_available():
    raise SystemExit("This release check requires a real NVIDIA GPU")
for required in ("mls_reference_input.npy", "mls_reference_predictions.npy"):
    if not (ARTIFACTS / required).is_file():
        raise SystemExit(f"Missing validation data: {ARTIFACTS / required}")

manager = manager_module.ModelManager(MLS)
if not manager.is_model_available():
    source = Path(os.environ.get("ALC_MLS_WEIGHTS",
                                 ROOT.parent / "model_release" / MLS.weights_filename))
    ok, message = manager.import_file(source)
    if not ok:
        raise SystemExit(f"Could not import the released weights: {message}")
print("weights:", manager.get_model_path(), flush=True)

xyz = np.load(ARTIFACTS / "mls_reference_input.npy")
reference_predictions = np.load(ARTIFACTS / "mls_reference_predictions.npy")
work = Path(tempfile.mkdtemp(prefix="alc_mls_gpu_"))
source_path = work / "mls_input.las"
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = [.00001, .00001, .00001]
header.offsets = np.floor(xyz.min(axis=0))
header.add_extra_dim(laspy.ExtraBytesParams(name="source_index", type="uint32"))
cloud = laspy.LasData(header)
cloud.x, cloud.y, cloud.z = xyz.T
n = len(xyz)
cloud.intensity = np.arange(n, dtype=np.uint16)
cloud.red = np.full(n, 1234, dtype=np.uint16)
cloud.green = np.full(n, 2345, dtype=np.uint16)
cloud.blue = np.full(n, 3456, dtype=np.uint16)
cloud.gps_time = np.arange(n) + 50000.5
cloud.classification = np.full(n, 9, dtype=np.uint8)
cloud.source_index = np.arange(n, dtype=np.uint32)
cloud.write(source_path)
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
original = laspy.read(source_path)
codes = {1: 2, 2: 3, 3: 5, 4: 6, 5: 200, 6: 201, 7: 202, 8: 14, 9: 1}
lookup = np.zeros(10, dtype=np.uint8)
for key, value in codes.items():
    lookup[key] = value
results = {}
outputs = {}

for mode in ("memory", "streaming"):
    start = time.monotonic()
    task = tasks.ClassificationTask(
        [source_path], work, "_" + mode, MLS.id,
        "cuda", "classification",
        tile_enabled=mode == "streaming", tile_auto=False,
        tile_size_m=5000, tile_buffer_m=0, tile_streaming=mode == "streaming",
        units_override="metre", output_codes=codes,
        prepare_qgis_view=True,
    )
    if not task.run():
        raise SystemExit(f"{mode} run failed: {task.error_message}")
    if len(task.output_files) != 1:
        raise SystemExit(f"{mode}: expected one output, got {task.output_files}")
    view = task.view_files[task.output_files[0]]
    if view == task.output_files[0] or not view.name.endswith(".qgis-view.copc.laz"):
        raise SystemExit(f"{mode}: no COPC viewing copy for an output with extra bytes")
    view_layer = QgsPointCloudLayer(str(view), view.stem, "copc")
    if not view_layer.isValid() or view_layer.dataProvider().pointCount() != n:
        raise SystemExit(f"{mode}: QGIS cannot open the viewing copy with {n} points")
    result = laspy.read(task.output_files[0])
    if len(result.points) != n:
        raise SystemExit(f"{mode}: {len(result.points)} points written for {n}")
    for name in ("X", "Y", "Z", "intensity", "red", "green", "blue", "gps_time", "source_index"):
        np.testing.assert_array_equal(result[name], original[name])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
        raise SystemExit("the input file was modified")
    records = [json.loads(bytes(v.record_data)) for v in result.header.vlrs
               if v.user_id == "AerialLiDAR"]
    if not records or records[-1]["model"] != MLS.id or records[-1]["classes"]["5"]["asprs"] != 200:
        raise SystemExit(f"{mode}: mapping VLR missing or wrong: {records}")
    if result.point_format.id != 7 or str(result.header.version) != "1.4":
        raise SystemExit(f"{mode}: expected a LAS 1.4 point format 7 upgrade")
    extra = {d.name for d in result.point_format.extra_dimensions}
    if extra != {"source_index"}:
        raise SystemExit(f"{mode}: extra dimensions changed: {extra}")
    outputs[mode] = np.asarray(result.classification).copy()
    if not set(np.unique(outputs[mode])) <= set(codes.values()):
        raise SystemExit(f"{mode}: codes outside the mapping: {np.unique(outputs[mode])}")
    results[mode] = {"points": n, "seconds": round(time.monotonic() - start, 2),
                     "point_format": result.point_format.id,
                     "classes": np.unique(outputs[mode]).tolist()}
    print("PASS", mode, results[mode], flush=True)

np.testing.assert_array_equal(outputs["memory"], outputs["streaming"])
litept = importlib.import_module(ROOT.name + ".core.backends.litept")
readback = laspy.read(source_path)
backend = litept.LitePTBackend(MLS.config_path, manager.get_model_path(), log=print)
backend.load("cuda")
try:
    direct = lookup[backend.predict(np.column_stack((readback.x, readback.y, readback.z)))]
finally:
    backend.unload()
direct_agreement = float((outputs["memory"] == direct).mean())
results["direct_same_las_agreement"] = direct_agreement
if direct_agreement < 0.9999:
    raise SystemExit(f"task output differs from direct inference: {direct_agreement:.4%}")
expected = lookup[reference_predictions]
agreement = float((outputs["memory"] == expected).mean())
results["reference_agreement"] = agreement
results["gpu"] = torch.cuda.get_device_name(0)
results["torch"] = torch.__version__
results["qgis"] = __import__("qgis.core", fromlist=["Qgis"]).Qgis.version()
print(json.dumps(results, indent=1), flush=True)
if agreement < 0.99:
    raise SystemExit(f"agreement with the reference predictions is {agreement:.4%}")
print(f"PASS real QGIS/CUDA classification: memory == streaming, source preserved, "
      f"{direct_agreement:.4%} identical to direct inference, "
      f"{agreement:.4%} to the reference predictions", flush=True)
