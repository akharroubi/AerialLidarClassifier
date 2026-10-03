"""Processing algorithm: classify airborne or Mobile Mapping LiDAR.

Wraps the same classification pipeline used by the interactive dock so
it can be invoked from the Processing Toolbox, the Graphical Modeler
and the ``qgis_process`` CLI.

Output is written to the standard ASPRS ``classification`` dimension by
default (LAS / LAZ 1.4 / point format 6 when codes require >5 bits).
"""

import logging
from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingOutputFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingUtils,
)

from ..config import PLUGIN_NAME, TILE_DEFAULT_BUFFER_M
from ..core.registry import MODELS
from ..utils.class_mapping import output_codes_json, output_model_spec, supports_output_codes, validate_model_output_field
from ..utils.compat import scoped_enum
from ..utils.las_units import UNIT_OVERRIDES, resolve_units

_LOG = logging.getLogger(__name__)

# Scoped enums (QGIS >= 3.36, required by QGIS 4 / PyQt6) with the
# QGIS 3.34 spellings as fallback.
try:
    from qgis.core import Qgis as _Qgis
    _FLAG_ADVANCED = _Qgis.ProcessingParameterFlag.Advanced
except AttributeError:
    _FLAG_ADVANCED = scoped_enum(QgsProcessingParameterDefinition, "Flag", "FlagAdvanced")
try:
    _NUMBER_DOUBLE = _Qgis.ProcessingNumberParameterType.Double
except AttributeError:
    _NUMBER_DOUBLE = scoped_enum(QgsProcessingParameterNumber, "Type", "Double")


class ClassifyLidarAlgorithm(QgsProcessingAlgorithm):
    """Classify a LAS / LAZ / COPC point cloud with a deep-learning model."""

    INPUT = "INPUT"
    MODEL = "MODEL"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    SUFFIX = "SUFFIX"
    FIELD_NAME = "FIELD_NAME"
    OUTPUT_CODES_JSON = "OUTPUT_CODES_JSON"
    DEVICE = "DEVICE"
    LOAD_AS_LAYER = "LOAD_AS_LAYER"
    TILE_ENABLED = "TILE_ENABLED"
    TILE_SIZE_M = "TILE_SIZE_M"
    TILE_BUFFER_M = "TILE_BUFFER_M"
    TILE_STREAMING = "TILE_STREAMING"
    UNITS = "UNITS"
    OUTPUT_FILE = "OUTPUT_FILE"

    # Index order is part of the algorithm's public interface (saved
    # models and scripts store the index), so new entries go at the end.
    DEVICE_OPTIONS = [
        "Auto (GPU if available)",
        "GPU (CUDA)",
        "CPU",
        "GPU (Apple MPS)",
    ]

    def tr(self, text: str) -> str:
        from .. import tr
        return tr(text, context="ClassifyLidarAlgorithm")

    def createInstance(self):  # noqa: N802
        return ClassifyLidarAlgorithm()

    def name(self) -> str:
        return "classify_lidar"

    def displayName(self) -> str:  # noqa: N802
        return self.tr("Classify airborne / Mobile Mapping LiDAR point cloud")

    def group(self) -> str:
        return self.tr("Classification")

    def groupId(self) -> str:  # noqa: N802
        return "classification"

    @staticmethod
    def _models_help() -> str:
        parts = []
        for spec in MODELS:
            rows = "".join(
                f"<tr><td>{info.model_id}</td><td>{info.name}</td><td>{info.asprs_code}</td></tr>"
                for info in spec.class_mapping.values()
            )
            parts.append(
                f"<p><b>{spec.display_name}</b> ({spec.device_requirement_text()}): "
                f"{spec.description} Licence: {spec.licence}.</p>"
                "<table><tr><th align=left>Model ID</th><th align=left>Class</th><th align=left>Default output code</th></tr>"
                f"{rows}</table>"
            )
        return "".join(parts)

    def shortHelpString(self) -> str:  # noqa: N802
        return self.tr(
            f"<h3>{PLUGIN_NAME}</h3>"
            "<p>Deep-learning semantic segmentation of airborne and Mobile Mapping "
            "point clouds (LAS / LAZ / COPC). Choose the model for the acquisition "
            "type. The Mobile Mapping model has nine classes and editable output codes.</p>"
            "<h4>Models</h4>"
            + self._models_help() +
            "<h4>ASPRS compliance</h4>"
            "<p>By default the classification is written to the standard "
            "LAS <code>classification</code> dimension. The file is "
            "automatically promoted to <b>LAS 1.4 / a compatible point format</b> "
            "when an assigned code exceeds the 5-bit legacy limit of point "
            "formats 0-5, so any classification code 0-255 is encoded losslessly. "
            "Mobile Mapping defaults 64, 65 and 66 are project-defined codes for "
            "pole-like objects, vehicles and fences/barriers.</p>"
            "<h4>Parameters</h4>"
            "<ul>"
            "<li><b>Input point cloud</b> - a single LAS, LAZ or "
            "<code>.copc.laz</code> file.</li>"
            "<li><b>Model</b> - LitePT-L Airborne or LitePT-L Mobile Mapping "
            "(NVIDIA CUDA GPU required), or SegFormer 3D Airborne (GPU or CPU). "
            "The weights download automatically the first time a model is used "
            "(SHA-256 verified).</li>"
            "<li><b>Output folder</b> - where the classified file is "
            "written.</li>"
            "<li><b>Output filename suffix</b> - appended before the "
            "extension. Default <code>_classified</code>.</li>"
            "<li><b>Classification field</b> - the dimension to write "
            "predictions into. Default <code>classification</code> "
            "(ASPRS standard). Mobile Mapping always writes to this existing field. "
            "Airborne models also support a new name to store raw model IDs in an extra-byte field.</li>"
            "<li><b>Mobile Mapping output codes (JSON)</b> - optional overrides keyed "
            "by model class ID, for example <code>{\"5\": 64, \"6\": 65, \"7\": 66}</code>. "
            "Use model IDs 1-9 from the Mobile Mapping table above. Values must be "
            "integers 0-255; omitted classes use model defaults. Repeated values "
            "merge classes. This parameter applies only to the Mobile Mapping model "
            "and the existing classification field. IDs 1-9 identify mapping rows; "
            "no extra label field is added. "
            "Processing uses this explicit parameter, independently of dock settings.</li>"
            "<li><b>Compute device</b> - <i>Auto</i> uses CUDA when "
            "PyTorch reports it, then Apple MPS, otherwise the CPU. "
            "<i>GPU (CUDA)</i> and <i>GPU (Apple MPS)</i> stop with a "
            "message when that device is not available.</li>"
            "<li><b>Load classified file in QGIS</b> - when on, the "
            "result is added to the project as a point-cloud layer.</li>"
            "</ul>"
            "<p>The output extension follows the input: <code>.las</code> "
            "in -> <code>.las</code> out, anything else -> <code>.laz</code>. "
            "COPC inputs are written as plain LAZ (the COPC spatial "
            "index is not regenerated).</p>"
            "<h4>Advanced parameters - tiling and streaming</h4>"
            "<p>Open the <b>Advanced Parameters</b> fold below to access:</p>"
            "<ul>"
            "<li><b>Process in spatial tiles</b> - splits the input into "
            "an N x N tile grid with a buffer halo, runs inference per "
            "tile and merges core predictions back. Recommended for files "
            "with many millions of points.</li>"
            "<li><b>Tile size in metres</b> - <code>0</code> means "
            "auto-size to ~10 M points per tile; positive value forces "
            "a square tile side.</li>"
            "<li><b>Tile buffer in metres</b> - context halo around "
            "each tile (50 m default). Predictions inside the buffer "
            "are discarded so points near tile edges still benefit from "
            "context.</li>"
            "<li><b>Streaming I/O</b> - process files larger than RAM. "
            "Reads the input chunk-by-chunk via <code>laspy</code>, stores "
            "per-tile points in disk-backed sidecars, runs per-tile "
            "inference one tile at a time and streams the output writer. "
            "Predictions and coverage use about 2 bytes/point during inference, plus tile/model workspace and one "
            "chunk. Requires tiling to be enabled.</li>"
            "<li><b>Input units</b> - the model works in metres. By "
            "default the unit is read from the file's CRS (WKT or GeoTIFF "
            "keys) and feet are converted before inference; the log says "
            "what was found. Force metres or feet when the header is "
            "missing or wrong.</li>"
            "</ul>"
            "<h4>Notes</h4>"
            "<ul>"
            "<li>Airborne model mappings retain their established defaults. "
            "Mobile Mapping output codes can be customized with the JSON parameter. "
            "The actual mapping and model identity are recorded in the output metadata.</li>"
            "</ul>")

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT,
                self.tr("Input point cloud (LAS / LAZ / COPC)"),
                extension="laz",
                fileFilter="Point clouds (*.las *.laz *.copc.laz)",
            )
        )
        # Index order is part of the public interface (saved models store
        # the index): keep the registry order stable.
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODEL,
                self.tr("Model"),
                options=[spec.display_name for spec in MODELS],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                self.tr("Output folder"),
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.SUFFIX,
                self.tr("Output filename suffix"),
                defaultValue="_classified",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEVICE,
                self.tr("Compute device"),
                options=self.DEVICE_OPTIONS,
                defaultValue=0,
            )
        )

        # FIELD_NAME tucked under Advanced - 99% of users want the
        # standard ASPRS 'classification' dimension.
        self._add_advanced(
            QgsProcessingParameterString(
                self.FIELD_NAME,
                self.tr(
                    "Classification field (default = ASPRS standard "
                    "'classification' dimension)"
                ),
                defaultValue="classification",
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD_AS_LAYER,
                self.tr("Load classified file in QGIS when finished"),
                defaultValue=True,
            )
        )
        self._add_advanced(
            QgsProcessingParameterString(
                self.OUTPUT_CODES_JSON,
                self.tr("Mobile Mapping output codes (JSON; blank = model defaults)"),
                defaultValue="",
                optional=True,
            )
        )
        # -- Advanced parameters (collapsed in the dialog by default) --
        self._add_advanced(
            QgsProcessingParameterBoolean(
                self.TILE_ENABLED,
                self.tr(
                    "Process in spatial tiles (recommended for very large files)"
                ),
                defaultValue=False,
            )
        )
        self._add_advanced(
            QgsProcessingParameterNumber(
                self.TILE_SIZE_M,
                self.tr(
                    "Tile size in metres (0 = auto, ~10 M points per tile)"
                ),
                type=_NUMBER_DOUBLE,
                defaultValue=0.0,
                minValue=0.0,
                maxValue=50_000.0,
            )
        )
        self._add_advanced(
            QgsProcessingParameterNumber(
                self.TILE_BUFFER_M,
                self.tr("Tile buffer in metres"),
                type=_NUMBER_DOUBLE,
                defaultValue=TILE_DEFAULT_BUFFER_M,
                minValue=0.0,
                maxValue=5_000.0,
            )
        )
        self._add_advanced(
            QgsProcessingParameterBoolean(
                self.TILE_STREAMING,
                self.tr(
                    "Streaming I/O (handle files larger than RAM; "
                    "requires tiling)"
                ),
                defaultValue=False,
            )
        )
        self._add_advanced(
            QgsProcessingParameterEnum(
                self.UNITS,
                self.tr("Input units (the model works in metres)"),
                options=[label for _key, label in UNIT_OVERRIDES],
                defaultValue=0,
            )
        )
        # The classified file itself, so the Graphical Modeler can chain
        # it into a next step (v1.0.2 only returned the folder).
        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_FILE, self.tr("Classified point cloud")
            )
        )

    # ------------------------------------------------------------------
    def _add_advanced(self, param):
        """Add a parameter under the dialog's 'Advanced Parameters' fold."""
        param.setFlags(param.flags() | _FLAG_ADVANCED)
        self.addParameter(param)

    # ------------------------------------------------------------------
    def _guard_input_output_collision(
            self,
            input_path: Path,
            output_path: Path) -> None:
        """Raise if writing would overwrite the input file in place."""
        from ..utils.output_safety import validate_output_paths
        try:
            validate_output_paths([input_path], [output_path])
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        import numpy as np

        model_idx = self.parameterAsEnum(parameters, self.MODEL, context)
        if not 0 <= model_idx < len(MODELS):
            model_idx = 0
        try:
            spec = output_model_spec(
                MODELS[model_idx],
                self.parameterAsString(parameters, self.OUTPUT_CODES_JSON, context),
            )
            field_name = validate_model_output_field(
                spec, self.parameterAsString(parameters, self.FIELD_NAME, context)
            )
        except ValueError as exc:
            raise QgsProcessingException(self.tr(f"Invalid output settings: {exc}")) from exc

        from ..utils.venv_manager import (
            ensure_venv_packages_available,
            get_venv_status,
        )
        from ..utils.model_manager import ModelManager
        from ..utils.logger import log_info, log_error

        is_ready, status_msg = get_venv_status()
        if not is_ready:
            raise QgsProcessingException(
                self.tr(
                    f"{PLUGIN_NAME} dependencies are not installed yet.\n\n"
                    f"Reason: {status_msg}\n\n"
                    "To install them: open "
                    f"'Plugins -> {PLUGIN_NAME}' from the menu (or click "
                    "the toolbar icon). The Setup dock will appear with "
                    "an Install button - it downloads a portable Python, "
                    "uv and the wheels into an isolated virtual env "
                    "(no admin rights, no impact on QGIS's own Python).\n\n"
                    "After the setup completes, this algorithm will work."
                )
            )
        # Make the venv site-packages importable from this Python process.
        ensure_venv_packages_available()

        manager = ModelManager(spec)

        input_path = Path(
            self.parameterAsFile(parameters, self.INPUT, context)
        )
        output_folder = Path(
            self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        )
        suffix = self.parameterAsString(
            parameters, self.SUFFIX, context) or "_classified"
        device_choice = self.parameterAsEnum(parameters, self.DEVICE, context)
        load_as_layer = self.parameterAsBool(
            parameters, self.LOAD_AS_LAYER, context
        )
        tile_enabled = self.parameterAsBool(
            parameters, self.TILE_ENABLED, context
        )
        tile_size_m = float(self.parameterAsDouble(
            parameters, self.TILE_SIZE_M, context
        ))
        tile_buffer_m = float(self.parameterAsDouble(
            parameters, self.TILE_BUFFER_M, context
        ))
        tile_auto = tile_size_m <= 0.0
        tile_streaming = self.parameterAsBool(
            parameters, self.TILE_STREAMING, context
        )
        units_idx = self.parameterAsEnum(parameters, self.UNITS, context)
        if not 0 <= units_idx < len(UNIT_OVERRIDES):
            units_idx = 0
        units_override = UNIT_OVERRIDES[units_idx][0]

        output_folder.mkdir(parents=True, exist_ok=True)

        # Resolve the compute device ("cuda", "mps" or "cpu"). The venv
        # was made importable above.
        import torch
        cuda_available = torch.cuda.is_available()
        mps_attr = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_attr is not None and mps_attr.is_available())
        if device_choice == 1:
            if not cuda_available:
                raise QgsProcessingException(self.tr(
                    "CUDA was requested but no usable GPU is available."
                ))
            device = "cuda"
        elif device_choice == 3:
            if not mps_available:
                raise QgsProcessingException(self.tr(
                    "Apple MPS was requested but is not available in this "
                    "torch build."
                ))
            device = "mps"
        elif device_choice == 2:
            device = "cpu"
        else:
            device = "cuda" if cuda_available else (
                "mps" if mps_available else "cpu"
            )
        if not spec.supports_device(device):
            raise QgsProcessingException(self.tr(
                f"{spec.display_name} does not run on '{device}' "
                f"({spec.device_requirement_text()}). Choose another model "
                "or compute device."
            ))

        # First use of a model: download its weights here (SHA-256
        # verified), so Processing and qgis_process need no manual step.
        if not manager.is_model_available():
            feedback.pushInfo(self.tr(
                f"Downloading the {spec.display_name} weights (first use, "
                f"about {spec.weights_size_mb:.0f} MB)..."
            ))

            def download_progress(received, total):
                if total > 0:
                    feedback.setProgress(min(99.0, 100.0 * received / total))

            try:
                ok, msg = manager.ensure_available(
                    download_progress, feedback.isCanceled)
            except InterruptedError:
                raise QgsProcessingException(self.tr("Cancelled."))
            if not ok:
                raise QgsProcessingException(msg)
            feedback.setProgress(0)

        # Imports deferred until dependencies are confirmed
        import laspy
        from ..core.backends import create_backend
        from ..workers.classifier_task import (
            ASPRS_CLASSIFICATION_FIELD,
            _classify_tiled,
            _ensure_asprs_classification_capacity,
            _resolve_output_path,
            _strip_copc_vlrs,
            _write_las,
        )

        from ..utils.output_safety import (validate_label_field, read_complete, checked_predictions,
                                           label_values, assign_labels, add_label_metadata, add_extra_dim_preserving_raw,
                                           extra_dim_description)
        with laspy.open(str(input_path)) as reader:
            field_name = validate_label_field(reader.header.point_format, field_name, laspy)
        output_path = _resolve_output_path(input_path, output_folder, suffix)
        self._guard_input_output_collision(input_path, output_path)

        # Snapshot includes explicit MMS output code overrides, independent
        # of interactive dock settings. Both writers record it in the VLR.
        class_mapping = spec.class_mapping
        if supports_output_codes(spec):
            feedback.pushInfo(self.tr(f"Mobile Mapping output codes: {output_codes_json(spec)}"))

        backend = create_backend(
            spec, manager.get_model_path(), log=feedback.pushWarning,
        )
        feedback.pushInfo(self.tr(f"Model: {spec.display_name}"))
        backend_loaded = False
        def predict_fn(xyz_m, progress_cb):
            nonlocal backend_loaded
            if not backend_loaded:
                backend.load(device)
                backend_loaded = True
            return backend.predict(xyz_m, progress_cb, feedback.isCanceled)

        # ---- Streaming dispatch (writes the output directly) ----------------
        is_asprs_field = field_name.lower() == ASPRS_CLASSIFICATION_FIELD
        if tile_enabled and tile_streaming:
            from ..workers.streaming_classifier import streaming_tiled_classify

            output_path = _resolve_output_path(
                input_path, output_folder, suffix
            )
            self._guard_input_output_collision(input_path, output_path)
            if output_path.exists():
                feedback.pushWarning(self.tr(
                    f"Output {output_path.name} already exists - overwriting."
                ))

            def stream_progress(p):
                if feedback.isCanceled():
                    raise InterruptedError()
                feedback.setProgress(int(min(100, max(0, p))))

            feedback.pushInfo(self.tr(
                "Streaming + tiling enabled - processing without loading "
                "the entire file into RAM."
            ))
            try:
                written = streaming_tiled_classify(
                    input_path=input_path,
                    output_path=output_path,
                    predict_fn=predict_fn,
                    device=device,
                    class_mapping=class_mapping,
                    field_description=f"AI classification ({spec.display_name})",
                    model_spec=spec,
                    laspy_module=laspy,
                    progress_callback=stream_progress,
                    cancel_callback=feedback.isCanceled,
                    field_name=field_name,
                    tile_auto=tile_auto,
                    tile_size_m=None if tile_auto else tile_size_m,
                    buffer_m=tile_buffer_m,
                    info_callback=feedback.pushInfo,
                    warning_callback=feedback.pushWarning,
                    units_override=units_override,
                )
            except InterruptedError:
                raise QgsProcessingException(self.tr("Cancelled."))
            finally:
                backend.unload()

            if written is None:
                raise QgsProcessingException(
                    self.tr("Streaming classification returned no result.")
                )

            log_info(f"Streaming processing wrote {written}")
            if load_as_layer:
                self._register_output_layer(written, context, feedback, spec, field_name)
            return {
                self.OUTPUT_FOLDER: str(output_folder),
                self.OUTPUT_FILE: str(written),
            }

        feedback.pushInfo(self.tr(f"Loading {input_path.name}..."))
        with laspy.open(str(input_path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            las = read_complete(reader)

        if len(las.x) == 0:
            raise QgsProcessingException(
                self.tr("Input file has zero points - nothing to classify.")
            )

        units = resolve_units(las.header, units_override)
        if units.angular:
            raise QgsProcessingException(self.tr(
                "The input is in geographic coordinates (degrees). The model "
                "needs projected coordinates: reproject the file (for example "
                "to the local UTM zone) and run again."
            ))
        feedback.pushInfo(self.tr(f"Coordinate units: {units.describe()}"))
        pcd = units.apply(np.transpose(np.array([las.x, las.y, las.z])))

        def progress_cb(p):
            if feedback.isCanceled():
                raise InterruptedError()
            feedback.setProgress(min(100, max(0, int(p))))

        feedback.pushInfo(self.tr(
            f"Running {spec.display_name} on {device.upper()}..."
        ))
        try:
            if tile_enabled:
                feedback.pushInfo(self.tr(
                    "Tiling enabled "
                    f"(auto={tile_auto}, tile={tile_size_m or 'auto'} m, "
                    f"buffer={tile_buffer_m:.0f} m)."
                ))
                preds = _classify_tiled(
                    pcd, predict_fn, progress_cb, feedback.isCanceled,
                    auto=tile_auto,
                    tile_size_m=None if tile_auto else tile_size_m,
                    buffer_m=tile_buffer_m,
                )
            else:
                preds = predict_fn(pcd, progress_cb)
        except InterruptedError:
            raise QgsProcessingException(self.tr("Cancelled."))
        except Exception as exc:  # pragma: no cover - surfaced to user
            log_error(f"Inference failed: {exc}")
            raise QgsProcessingException(
                self.tr(f"Classification failed: {exc}")
            )
        finally:
            backend.unload()

        if preds is None:
            raise QgsProcessingException(
                self.tr("Classifier returned no results.")
            )

        preds = checked_predictions(preds, len(las.points))
        is_asprs_field = field_name == ASPRS_CLASSIFICATION_FIELD
        asprs = label_values(preds, class_mapping, raw_ids=not is_asprs_field)

        if is_asprs_field:
            try:
                max_code = int(asprs.max())
            except ValueError:
                max_code = 0
            las = _ensure_asprs_classification_capacity(
                las, laspy, max_code,
                preserve_legacy_angle=not supports_output_codes(spec))
            las.classification = asprs.astype(np.uint8)
            feedback.pushInfo(
                self.tr(
                    "Writing to standard ASPRS 'classification' dimension "
                    f"(point format {las.point_format.id}, "
                    f"LAS {las.header.version})."
                )
            )
        else:
            if field_name in las.point_format.dimension_names:
                assign_labels(las, field_name, asprs)
            else:
                add_extra_dim_preserving_raw(las, laspy.ExtraBytesParams(
                    name=field_name, type="int32",
                    description=extra_dim_description(f"AI classification ({spec.display_name})"),
                ))
                assign_labels(las, field_name, asprs)
            feedback.pushInfo(
                self.tr(
                    f"Writing to extra-byte field '{field_name}' "
                    "(non-standard, not ASPRS-compliant)."
                )
            )

        add_label_metadata(las.header, spec, field_name)
        _strip_copc_vlrs(las)

        output_path = _resolve_output_path(input_path, output_folder, suffix)
        self._guard_input_output_collision(input_path, output_path)
        if output_path.exists():
            feedback.pushWarning(self.tr(
                f"Output {output_path.name} already exists - overwriting."
            ))
        feedback.pushInfo(self.tr(f"Writing {output_path.name}..."))
        _write_las(las, output_path, laspy)

        log_info(f"Processing algorithm wrote {output_path}")

        if load_as_layer:
            self._register_output_layer(output_path, context, feedback, spec, field_name)

        return {
            self.OUTPUT_FOLDER: str(output_folder),
            self.OUTPUT_FILE: str(output_path),
        }

    # ------------------------------------------------------------------
    def _register_output_layer(self, output_path: Path, context, feedback, spec=None,
                               field_name="classification") -> None:
        """Ask Processing to load the result once the algorithm has finished.

        ``processAlgorithm`` runs in a worker thread when launched from
        the Toolbox, and creating a layer plus adding it to the project
        from there is not thread-safe (v1.0.2 did exactly that and could
        crash QGIS). ``addLayerToLoadOnCompletion`` defers the load to
        the main thread; the post-processor then attaches the 3D
        renderer there, so a 3D Map View shows the points instead of a
        flat sprite.
        """
        project = context.project()
        if project is None:
            feedback.pushInfo(self.tr(
                "No project in this context; the classified file is not "
                "loaded as a layer."
            ))
            return

        # This method runs in the Processing worker. Keep the primary output
        # as OUTPUT_FILE and register only the optional compatible viewing copy.
        from ..utils.pointcloud_view import prepare_qgis_view
        view_path = prepare_qgis_view(
            output_path, cancel_callback=feedback.isCanceled,
            info_callback=feedback.pushInfo, warning_callback=feedback.pushWarning,
        )
        if feedback.isCanceled():
            return

        details = QgsProcessingContext.LayerDetails(
            output_path.stem, project, self.OUTPUT_FILE, _POINT_CLOUD_HINT
        )
        # Processing only keeps a weak reference to the post-processor;
        # one that is garbage-collected is silently skipped.
        standard_field = spec is not None and field_name.strip().lower() == "classification"
        self._post_processor = _PointCloud3DPostProcessor(
            spec.class_mapping if standard_field else None,
            classify_2d=bool(standard_field and supports_output_codes(spec)),
            primary_output=output_path,
        )
        details.setPostProcessor(self._post_processor)
        context.addLayerToLoadOnCompletion(str(view_path), details)
        feedback.pushInfo(self.tr(
            f"'{view_path.name}' will be added to the project when the "
            "algorithm finishes."
        ))


# Point-cloud hint for addLayerToLoadOnCompletion (QGIS >= 3.22); older
# builds fall back to the generic loader, which also handles LAS/LAZ.
_POINT_CLOUD_HINT = getattr(
    QgsProcessingUtils.LayerHint, "PointCloud",
    QgsProcessingUtils.LayerHint.UnknownType,
)


class _PointCloud3DPostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Runs on the main thread after Processing loaded the output layer."""

    def __init__(self, class_mapping=None, classify_2d=False, primary_output=None):
        super().__init__()
        from copy import deepcopy
        self.class_mapping = deepcopy(class_mapping)
        self.classify_2d = classify_2d
        self.primary_output = primary_output

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            from ..utils.helpers import enable_point_cloud_3d_rendering
            enable_point_cloud_3d_rendering(layer, self.class_mapping, self.classify_2d)
        except Exception as exc:
            feedback.pushWarning(
                f"Could not attach a 3D renderer to '{layer.name()}': {exc}"
            )
        if self.primary_output is not None:
            # Viewing copies of earlier runs that no layer uses any more.
            try:
                from ..utils.pointcloud_view import remove_stale_views
                project = context.project() if context is not None else None
                in_use = ([lyr.source() for lyr in project.mapLayers().values()]
                          if project is not None else [])
                remove_stale_views(self.primary_output, keep=layer.source(), in_use=in_use)
            except Exception:
                _LOG.debug("Ignored non-fatal error", exc_info=True)
