"""Release acceptance test of the built ZIP in fresh QGIS 3 and 4 profiles.

    python tests/e2e_mls_qgis_process.py --zip dist/lidar_ai_classifier_v1.2.0.zip \
        --input <reference LAS> --work <scratch parent> [--qgis3 BAT] [--qgis4 BAT]

Both runtimes are required. Each run gets a new private directory below
--work; existing files and profiles are never removed. The test serves the
released weights on localhost, checks the real first-use download and its
SHA-256, then runs MODEL=2 on CUDA with default codes in memory (QGIS 3)
and custom codes in streaming mode (QGIS 4). It verifies file integrity and
cross-runtime class IDs. No reference-label field is required.

The plugin environment comes from AERIAL_LIDAR_CLASSIFIER_CACHE_DIR (an
existing install). Reports and logs remain in the printed work directory.
Any missing prerequisite, integrity failure or missing run returns exit 1.
"""
import argparse
import ast
import functools
import hashlib
import http.server
import json
import os
import subprocess  # nosec B404 - developer tool, not shipped in the plugin ZIP
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath

import laspy
import numpy as np

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent
NAME = "Aerial_LiDAR_Classifier"
WEIGHTS = "litept_l_mls_5cm_fp16.pth"
DEFAULT_CODES = {1: 2, 2: 3, 3: 5, 4: 6, 5: 64, 6: 65, 7: 66, 8: 14, 9: 1}
CUSTOM_CODES = {**DEFAULT_CODES, 5: 200, 6: 201, 7: 255}
MIN_AGREEMENT = 0.9995


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_weights_hash(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        source = archive.read(f"{NAME}/core/registry.py").decode("utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if any(isinstance(target, ast.Name) and target.id == "LITEPT_L_MLS"
               for target in node.targets):
            fields = {item.arg: item.value for item in node.value.keywords}
            if ast.literal_eval(fields["weights_filename"]) != WEIGHTS:
                raise ValueError("The ZIP specifies a different mobile mapping weights file")
            value = ast.literal_eval(fields["weights_sha256"])
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid mobile mapping SHA-256 in the ZIP")
            return value
    raise ValueError("Mobile mapping model not found in the ZIP registry")


def serve(folder):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802 - HTTP handler API
            self.server.weight_requests.append(self.path)
            super().do_GET()

    handler = functools.partial(Handler, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.weight_requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def make_profile(root, zip_path, ini_name, weights_url):
    # Refuse reuse. Only a newly created private run directory is writable.
    root.mkdir(parents=False, exist_ok=False)
    plugins = root / "profiles" / "default" / "python" / "plugins"
    plugins.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if (not parts or parts[0] != NAME or ".." in parts
                    or "\\" in member.filename or ":" in member.filename):
                raise ValueError(f"Unsafe plugin ZIP entry: {member.filename}")
        archive.extractall(plugins)
    ini = root / "profiles" / "default" / "QGIS" / ini_name
    ini.parent.mkdir(parents=True)
    ini.write_text(
        "[PythonPlugins]\n"
        f"{NAME}=true\n\n"
        "[AerialLidarClassifier]\n"
        f"model_url\\litept_l_mls_5cm={weights_url}\n",
        encoding="utf-8")


def run_qgis_process(launcher, profile, cache, args, log):
    env = dict(os.environ, QGIS_CUSTOM_CONFIG_PATH=str(profile),
               AERIAL_LIDAR_CLASSIFIER_CACHE_DIR=str(cache))
    command = ["cmd", "/c", str(launcher), "--json", "run",
               "aeriallidar:classify_lidar", "--", *args]
    start = time.monotonic()
    with open(log, "w", encoding="utf-8", errors="replace") as stream:
        code = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,  # nosec B603
                              check=False, timeout=1800).returncode
    return code, time.monotonic() - start


def record_signature(records):
    # laspy may reorder the Extra Bytes VLR during a format upgrade. Compare
    # every record's raw payload and metadata, retaining duplicate counts.
    return sorted((record.user_id, record.record_id, record.description,
                   bytes(record.record_data_bytes()))
                  for record in (records or []) if record.user_id != "AerialLiDAR")


def check(source_path, out_path, codes):
    source, output = laspy.read(source_path), laspy.read(out_path)
    problems = []
    if len(output.points) != len(source.points):
        problems.append(f"{len(output.points)} points written for {len(source.points)}")
    for attr in ("scales", "offsets", "mins", "maxs"):
        if not np.array_equal(getattr(source.header, attr), getattr(output.header, attr)):
            problems.append(f"header {attr} changed")
    if source.header.parse_crs() != output.header.parse_crs():
        problems.append("CRS changed")
    for attr in ("vlrs", "evlrs"):
        if record_signature(getattr(source.header, attr)) != record_signature(getattr(output.header, attr)):
            problems.append(f"non-plugin {attr.upper()} changed")
    source_extras = list(source.point_format.extra_dimension_names)
    output_extras = list(output.point_format.extra_dimension_names)
    if output_extras != source_extras:
        problems.append(f"extra dimensions changed: {source_extras} -> {output_extras}")
    upgrade = source.point_format.id < 6 and max(codes.values()) > 31
    expected_format = ({0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 10}[source.point_format.id]
                       if upgrade else source.point_format.id)
    expected_version = "1.4" if upgrade else str(source.header.version)
    if output.point_format.id != expected_format or str(output.header.version) != expected_version:
        problems.append("unexpected LAS version or point-format conversion")
    output_names = set(output.point_format.dimension_names)
    for name in source.point_format.dimension_names:
        if name == "classification":
            continue
        if name == "scan_angle_rank" and upgrade:
            if "scan_angle" not in output_names:
                problems.append("converted scan angle missing")
            else:
                original = np.asarray(source[name], dtype=np.float64)
                converted = np.asarray(output.scan_angle, dtype=np.float64) * 0.006
                if not np.all(np.abs(original - converted) <= 0.003000001):
                    problems.append("legacy scan angle changed by more than 0.003 degrees")
            continue
        if name not in output_names:
            problems.append(f"attribute {name} missing")
            continue
        if name in source_extras:
            left, right = source.points.array[name], output.points.array[name]
            same = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        else:
            left, right = np.asarray(source[name]), np.asarray(output[name])
            same = left.shape == right.shape and left.tobytes() == right.tobytes()
        if not same:
            problems.append(f"attribute {name} changed")
    labels = np.asarray(output.classification).copy()
    present = set(np.unique(labels).tolist())
    if not present <= set(codes.values()):
        problems.append(f"unexpected codes {sorted(present - set(codes.values()))}")
    records = [json.loads(bytes(record.record_data_bytes())) for record in output.header.vlrs
               if record.user_id == "AerialLiDAR" and record.record_id == 1]
    mappings = [record for record in records if record.get("field") == "classification"]
    if len(mappings) != 1 or mappings[0].get("model") != "litept_l_mls_5cm":
        problems.append("classification provenance VLR missing or ambiguous")
    else:
        written = {int(k): value["asprs"] for k, value in mappings[0]["classes"].items()}
        if written != codes:
            problems.append(f"VLR mapping {written} != {codes}")
    return problems, labels


def class_ids(labels, codes):
    lookup = np.full(256, -1, dtype=np.int16)
    for class_id, code in codes.items():
        lookup[code] = class_id
    result = lookup[labels]
    if np.any(result < 0):
        raise ValueError("Cannot compare output containing unmapped codes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--weights-dir", default=str(PLUGIN.parent / "model_release"))
    parser.add_argument("--cache", default=os.environ.get("AERIAL_LIDAR_CLASSIFIER_CACHE_DIR",
                                                          str(Path.home() / ".qgis_aerial_lidar_classifier")))
    parser.add_argument("--qgis3", default=r"C:\Program Files\QGIS 3.44.10\bin\qgis_process-qgis-ltr.bat")
    parser.add_argument("--qgis4", default=r"C:\Users\user\qgis4_test\app\bin\qgis_process-qgis.bat")
    args = parser.parse_args()
    work_parent = Path(args.work).resolve()
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="qgis_e2e_", dir=work_parent)).resolve()
    if work.parent != work_parent:
        raise RuntimeError("Private work directory escaped the requested parent")
    report_path = work / "e2e_report.json"
    report = {"work_dir": str(work), "report": str(report_path), "input": str(Path(args.input).resolve()),
              "zip": str(Path(args.zip).resolve()), "passed": False}
    print(f"Work directory: {work}", flush=True)
    server, source_hash = None, None
    labels = {}
    failed = False
    try:
        launchers = {"qgis3": Path(args.qgis3).resolve(), "qgis4": Path(args.qgis4).resolve()}
        missing = [str(path) for path in launchers.values() if not path.is_file()]
        if missing:
            raise ValueError("Both QGIS launchers are required; missing: " + "; ".join(missing))
        source_path = Path(args.input).resolve(strict=True)
        zip_path = Path(args.zip).resolve(strict=True)
        weights_dir = Path(args.weights_dir).resolve(strict=True)
        expected_hash = release_weights_hash(zip_path)
        if sha256(weights_dir / WEIGHTS) != expected_hash:
            raise ValueError("Release weights do not match the built ZIP's SHA-256")
        source_hash = sha256(source_path)
        report.update(input_sha256=source_hash, zip_sha256=sha256(zip_path), weights_sha256=expected_hash)
        with laspy.open(source_path) as reader:
            if not reader.header.point_count:
                raise ValueError("A non-empty reference LAS is required")
            # A single streaming tile preserves the same inference context.
            span = float(np.max(reader.header.maxs[:2] - reader.header.mins[:2]))
            if not np.isfinite(span):
                raise ValueError("Reference LAS has non-finite bounds")
        tile_size = max(1.0, span * 2.0 + 1.0)
        server = serve(weights_dir)
        url = f"http://127.0.0.1:{server.server_address[1]}/{WEIGHTS}"
        runs = (
            ("qgis3", "QGIS3.ini", DEFAULT_CODES, ["TILE_ENABLED=false"]),
            ("qgis4", "QGIS4.ini", CUSTOM_CODES,
             ["TILE_ENABLED=true", "TILE_STREAMING=true", f"TILE_SIZE_M={tile_size}", "TILE_BUFFER_M=0"]),
        )
        for label, ini, codes, extra in runs:
            profile, out_dir = work / f"profile_{label}", work / f"out_{label}"
            entry = {"launcher": str(launchers[label]), "log": str(work / f"{label}.log")}
            try:
                make_profile(profile, zip_path, ini, url)
                params = [f"INPUT={source_path}", f"OUTPUT_FOLDER={out_dir}", "MODEL=2", "DEVICE=1",
                          "LOAD_AS_LAYER=false", "UNITS=0", *extra]
                if codes != DEFAULT_CODES:
                    params.append("OUTPUT_CODES_JSON=" + json.dumps({str(k): v for k, v in codes.items()},
                                                                      separators=(",", ":")))
                requests_before = len(server.weight_requests)
                code, seconds = run_qgis_process(launchers[label], profile, Path(args.cache).resolve(),
                                                 params, work / f"{label}.log")
                entry.update(exit_code=code, seconds=round(seconds, 1))
                if code:
                    raise ValueError(f"qgis_process exited with status {code}; see {entry['log']}")
                cached = profile / "profiles" / "default" / "AerialLidarClassifier" / "models" / "litept_l_mls_5cm" / WEIGHTS
                downloaded = f"/{WEIGHTS}" in server.weight_requests[requests_before:]
                entry["weights_downloaded_from_localhost"] = downloaded
                if not downloaded or not cached.is_file() or sha256(cached) != expected_hash:
                    raise ValueError("Fresh-profile localhost download or cached SHA-256 verification failed")
                entry["cached_weights_sha256"] = expected_hash
                outputs = sorted(out_dir.glob("*_classified.la*")) if out_dir.exists() else []
                if len(outputs) != 1:
                    raise ValueError(f"Expected one classified output, found {len(outputs)}")
                problems, labels[label] = check(source_path, outputs[0], codes)
                entry.update(output=str(outputs[0]), points=len(labels[label]), problems=problems)
                if problems:
                    failed = True
                if sha256(source_path) != source_hash:
                    raise ValueError("The source input file was modified")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failed = True
            report[label] = entry
            print(label, json.dumps(entry), flush=True)
        if set(labels) != {"qgis3", "qgis4"}:
            raise ValueError("Both QGIS versions must produce validated outputs")
        ids3, ids4 = class_ids(labels["qgis3"], DEFAULT_CODES), class_ids(labels["qgis4"], CUSTOM_CODES)
        if ids3.shape != ids4.shape:
            raise ValueError("Cross-runtime output lengths differ")
        agreement = float(np.mean(ids3 == ids4))
        report.update(qgis3_vs_qgis4_label_agreement=agreement, minimum_agreement=MIN_AGREEMENT)
        if agreement < MIN_AGREEMENT:
            raise ValueError(f"Cross-runtime class agreement {agreement:.6f} is below {MIN_AGREEMENT}")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if source_hash is not None:
            try:
                report["input_sha256_after"] = sha256(Path(args.input).resolve())
                if report["input_sha256_after"] != source_hash:
                    report["source_changed"] = True
                    failed = True
            except OSError as exc:
                report["source_check_error"] = str(exc)
                failed = True
    report["passed"] = not failed
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    print("E2E FAILED" if failed else "E2E PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
