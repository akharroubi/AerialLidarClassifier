"""Shared, fail-closed LAS output validation and atomic publication."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

import numpy as np


def paths_alias(left, right):
    left, right = Path(left), Path(right)
    if os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve())):
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def validate_output_paths(inputs, outputs):
    """Check the entire batch, including symlinks and existing hard links."""
    inputs, outputs = list(inputs), list(outputs)
    for i, output in enumerate(outputs):
        for source in inputs:
            if paths_alias(source, output):
                raise ValueError(f"Output would overwrite an input: {output}. Choose another folder or suffix.")
        for previous in outputs[:i]:
            if paths_alias(previous, output):
                raise ValueError(f"Two inputs have the same output: {output}. Use separate output folders.")


@contextmanager
def atomic_output_path(destination):
    """Publish only a successfully closed output, on the same filesystem."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.stem}.partial-",
                                suffix=destination.suffix, dir=str(destination.parent))
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(str(temporary), str(destination))
    finally:
        temporary.unlink(missing_ok=True)


def validate_label_field(point_format, name, laspy_module):
    """Allow classification or an unscaled scalar integer extra dimension."""
    name = (name or "classification").strip() or "classification"
    if name.lower() == "classification":
        return "classification"
    if "\x00" in name or len(name.encode("utf-8")) > 32:
        raise ValueError("The classification field name must fit in 32 UTF-8 bytes and contain no NUL.")
    standard = {dim.lower() for i in range(11)
                for dim in laspy_module.PointFormat(i).standard_dimension_names}
    if name.lower() in standard or name.lower() in {"x", "y", "z"}:
        raise ValueError(f"'{name}' is a protected LAS dimension. Choose a new extra field name.")
    for dim in point_format.extra_dimensions:
        if dim.name.lower() != name.lower():
            continue
        if dim.name != name:
            raise ValueError(f"Field '{dim.name}' already exists with different capitalization.")
        if (dim.num_elements != 1 or dim.dtype.kind not in "iu"
                or dim.scales is not None or dim.offsets is not None):
            raise ValueError(f"Existing field '{name}' must be an unscaled scalar integer classification field.")
    return name


def checked_predictions(predictions, count):
    predictions = np.asarray(predictions)
    if predictions.shape != (count,) or predictions.dtype.kind not in "iu":
        raise ValueError(f"Model must return one integer label per point ({count:,} expected).")
    return predictions


def label_values(predictions, mapping, raw_ids=False):
    """Reject incomplete predictions instead of publishing implicit zeros."""
    missing = ~np.isin(predictions, list(mapping))
    if missing.any():
        ids = np.unique(predictions[missing]).tolist()
        raise ValueError(f"Model returned unmapped IDs {ids} for {int(missing.sum()):,} points; no output published.")
    if raw_ids:
        values = predictions
    else:
        values = np.zeros(len(predictions), dtype=np.int32)
        for key, info in mapping.items():
            values[predictions == key] = int(info.asprs_code)
    if values.size and (values.min() < 0 or values.max() > 255):
        raise ValueError("Classification labels must be in the range 0–255.")
    return values.astype(np.uint8)


def assign_labels(record, name, values):
    if name != "classification":
        dtype = record.point_format.dimension_by_name(name).dtype
        limits = np.iinfo(dtype)
        if values.size and (values.min() < limits.min or values.max() > limits.max):
            raise ValueError(f"Labels do not fit existing field '{name}' ({dtype}).")
    record[name] = values


def add_extra_dim_preserving_raw(las, params):
    """laspy's generic dimension copy round-trips scaled extras through floats."""
    original = las.points.array
    las.add_extra_dim(params)
    for name in original.dtype.names:
        las.points.array[name] = original[name]


def read_complete(reader):
    validate_waveform_storage(reader.header)
    declared = int(reader.header.point_count)
    result = reader.read()
    if len(result.points) != declared:
        raise ValueError(f"Truncated LAS/LAZ: header declares {declared:,} points, read {len(result.points):,}.")
    return result


def validate_waveform_storage(header):
    """Packet payload relocation is not implemented; never publish dangling offsets."""
    encoding = header.global_encoding
    if (getattr(header, "start_of_waveform_data_packet_record", 0)
            or getattr(encoding, "waveform_data_packets_internal", False)
            or getattr(encoding, "waveform_data_packets_external", False)):
        raise ValueError("Waveform packet payloads are not supported by this writer. Export a point-only LAS/LAZ copy before classification; the source is unchanged.")


def upgraded_point_format(old_id):
    return {0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 10}.get(old_id, old_id)


def convert_points_preserving_fields(points, point_format, scales, offsets, laspy_module):
    """Convert packed layouts, retaining raw extra bytes and legacy scan angles."""
    converted = laspy_module.PackedPointRecord.from_point_record(points, point_format)
    result = laspy_module.ScaleAwarePointRecord(converted.array, point_format, scales, offsets)
    for dim in points.point_format.extra_dimensions:
        result.array[dim.name] = points.array[dim.name]
    if points.point_format.id < 6 <= point_format.id:
        old_angle = np.asarray(points.scan_angle_rank)
        result.scan_angle = np.rint(old_angle.astype(np.float64) / 0.006).astype(np.int16)
        result.array["scan_angle_rank"] = old_angle
    return result


def upgrade_header_preserving_fields(header, laspy_module):
    from copy import deepcopy
    temporary = deepcopy(header)
    temporary.point_count = 0
    upgraded = laspy_module.convert(laspy_module.LasData(temporary),
                                    point_format_id=upgraded_point_format(header.point_format.id),
                                    file_version="1.4").header
    if header.point_format.id < 6:
        upgraded.add_extra_dim(laspy_module.ExtraBytesParams(
            name="scan_angle_rank", type="int8", description="Original legacy scan angle"))
    return upgraded


def add_label_metadata(header, spec, field):
    import json
    import laspy
    data = {"model": spec.id, "weights_sha256": spec.weights_sha256,
            "field": field, "encoding": "ASPRS" if field == "classification" else "raw_model_ids",
            "classes": {str(k): {"name": v.name, "asprs": v.asprs_code}
                        for k, v in spec.class_mapping.items()}}
    # Re-classifying a file replaces the record of the same field instead
    # of stacking stale ones; records of other fields stay.
    for index in range(len(header.vlrs) - 1, -1, -1):
        vlr = header.vlrs[index]
        if getattr(vlr, "user_id", "") != "AerialLiDAR" or vlr.record_id != 1:
            continue
        try:
            previous = json.loads(bytes(vlr.record_data).decode("utf-8"))
        except Exception:
            previous = {}
        if previous.get("field", field) == field:
            del header.vlrs[index]
    header.vlrs.append(laspy.VLR(user_id="AerialLiDAR", record_id=1,
                               description="Classification provenance",
                               record_data=json.dumps(data, ensure_ascii=True).encode("utf-8")))
