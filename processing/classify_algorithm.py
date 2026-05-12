"""Processing algorithm: classify aerial LiDAR point cloud.

Wraps the same classification pipeline used by the interactive dock so
it can be invoked from the Processing Toolbox, the Graphical Modeler
and the ``qgis_process`` CLI.

Output is written to the standard ASPRS ``classification`` dimension by
default (LAS / LAZ 1.4 / point format 6 when codes require >5 bits).
"""

from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from ..config import (
    DEFAULT_CLASS_MAPPING,
    PLUGIN_NAME,
    TILE_DEFAULT_BUFFER_M,
)


class ClassifyLidarAlgorithm(QgsProcessingAlgorithm):
    """Classify a LAS / LAZ / COPC point cloud with the 3D SegFormer model."""

    INPUT = "INPUT"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    SUFFIX = "SUFFIX"
    FIELD_NAME = "FIELD_NAME"
    DEVICE = "DEVICE"
    LOAD_AS_LAYER = "LOAD_AS_LAYER"
    TILE_ENABLED = "TILE_ENABLED"
    TILE_SIZE_M = "TILE_SIZE_M"
    TILE_BUFFER_M = "TILE_BUFFER_M"
    TILE_STREAMING = "TILE_STREAMING"

    DEVICE_OPTIONS = ["Auto (GPU if available)", "GPU (CUDA)", "CPU"]

    def tr(self, text: str) -> str:
        from .. import tr
        return tr(text, context="ClassifyLidarAlgorithm")

    def createInstance(self):  # noqa: N802
        return ClassifyLidarAlgorithm()

    def name(self) -> str:
        return "classify_lidar"

    def displayName(self) -> str:  # noqa: N802
        return self.tr("Classify aerial LiDAR point cloud")

    def group(self) -> str:
        return self.tr("Classification")

    def groupId(self) -> str:  # noqa: N802
        return "classification"

    def shortHelpString(self) -> str:  # noqa: N802
        return self.tr(
            "<h3>Aerial LiDAR Classifier</h3>"
            "<p>Deep-learning semantic segmentation of aerial LiDAR point "
            "clouds (LAS / LAZ / COPC) using the 3D SegFormer "
            "<b>UrbanFiltering</b> model from the "
            "<a href=\"https://github.com/NRCan/TreeAIBox\">TreeAIBox</a> "
            "project (Z. Xi - NRCan, &copy; Crown Copyright, CC BY-NC 4.0).</p>"
            "<h4>Output classes</h4>"
            "<p>The model produces five primary classes mapped to standard "
            "ASPRS LAS 1.4 codes, plus two auxiliary classes routed to "
            "ASPRS 1 = <i>Unclassified</i> by default:</p>"
            "<table>"
            "<tr><th align=left>Class</th><th align=left>ASPRS code</th></tr>"
            "<tr><td>Ground</td><td>2</td></tr>"
            "<tr><td>Vegetation</td><td>5</td></tr>"
            "<tr><td>Building</td><td>6</td></tr>"
            "<tr><td>Wires (powerlines)</td><td>14</td></tr>"
            "<tr><td>Pole</td><td>15</td></tr>"
            "<tr><td>Vehicles <i>(aux.)</i></td><td>1 - Unclassified</td></tr>"
            "<tr><td>Fence <i>(aux.)</i></td><td>1 - Unclassified</td></tr>"
            "</table>"
            "<h4>ASPRS compliance</h4>"
            "<p>By default the classification is written to the standard "
            "LAS <code>classification</code> dimension. The file is "
            "automatically promoted to <b>LAS 1.4 / point format 6</b> "
            "when an assigned code exceeds the 5-bit legacy limit of point "
            "formats 0-5, so any ASPRS code 0-255 is encoded losslessly.</p>"
            "<h4>Parameters</h4>"
            "<ul>"
            "<li><b>Input point cloud</b> - a single LAS, LAZ or "
            "<code>.copc.laz</code> file.</li>"
            "<li><b>Output folder</b> - where the classified file is "
            "written.</li>"
            "<li><b>Output filename suffix</b> - appended before the "
            "extension. Default <code>_classified</code>.</li>"
            "<li><b>Classification field</b> - the dimension to write "
            "predictions into. Default <code>classification</code> "
            "(ASPRS standard). Use any other name to add an extra-byte "
            "field instead.</li>"
            "<li><b>Compute device</b> - <i>Auto</i> uses the GPU when "
            "PyTorch reports CUDA available; otherwise falls back to CPU."
            "</li>"
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
            "<li><b>Tile size in CRS units</b> - <code>0</code> means "
            "auto-size to ~10 M points per tile; positive value forces "
            "a square tile side.</li>"
            "<li><b>Tile buffer in CRS units</b> - context halo around "
            "each tile (50 m default). Predictions inside the buffer "
            "are discarded so points near tile edges still benefit from "
            "context.</li>"
            "<li><b>Streaming I/O</b> - process files larger than RAM. "
            "Reads the input chunk-by-chunk via <code>laspy</code>, stores "
            "per-tile points in disk-backed sidecars, runs per-tile "
            "inference one tile at a time and streams the output writer. "
            "Memory footprint is roughly 4 bytes/point + one tile + one "
            "chunk. Requires tiling to be enabled.</li>"
            "</ul>"
            "<h4>Notes</h4>"
            "<ul>"
            "<li>The class mapping is internal and not user-editable - the "
            "output is always ASPRS-compliant by construction. The five "
            "primary classes map to their standard ASPRS codes; Vehicles "
            "and Fences map to <i>Unclassified</i> (ASPRS 1).</li>"
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
                    "Tile size in CRS units (0 = auto, ~10 M points per tile)"
                ),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                maxValue=50_000.0,
            )
        )
        self._add_advanced(
            QgsProcessingParameterNumber(
                self.TILE_BUFFER_M,
                self.tr("Tile buffer in CRS units"),
                type=QgsProcessingParameterNumber.Double,
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

    # ------------------------------------------------------------------
    def _add_advanced(self, param):
        """Add a parameter under the dialog's 'Advanced Parameters' fold."""
        param.setFlags(
            param.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(param)

    # ------------------------------------------------------------------
    def _guard_input_output_collision(
            self,
            input_path: Path,
            output_path: Path) -> None:
        """Raise if writing would overwrite the input file in place."""
        try:
            same = input_path.resolve() == output_path.resolve()
        except Exception:
            same = False
        if same:
            raise QgsProcessingException(self.tr(
                "Refusing to overwrite the input file. Choose a different "
                "output folder or filename suffix."
            ))

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        import numpy as np

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

        if not ModelManager.is_model_available():
            raise QgsProcessingException(
                self.tr(
                    "Model weights are not downloaded yet. Open "
                    f"'{PLUGIN_NAME}' from the Plugins menu and click the "
                    "download icon in the panel header first."
                )
            )

        input_path = Path(
            self.parameterAsFile(parameters, self.INPUT, context)
        )
        output_folder = Path(
            self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        )
        suffix = self.parameterAsString(
            parameters, self.SUFFIX, context) or "_classified"
        field_name = (
            self.parameterAsString(parameters, self.FIELD_NAME, context)
            or "classification"
        )
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

        output_folder.mkdir(parents=True, exist_ok=True)

        # Resolve compute device. The venv was made importable above.
        import torch
        cuda_available = torch.cuda.is_available()
        if device_choice == 1 and not cuda_available:
            raise QgsProcessingException(
                self.tr("CUDA was requested but no usable GPU is available.")
            )
        use_cuda = (
            device_choice == 0 and cuda_available) or device_choice == 1

        # Imports deferred until dependencies are confirmed
        import laspy
        from ..core.classifier_core import filterPoints
        from ..workers.classifier_task import (
            ASPRS_CLASSIFICATION_FIELD,
            _classify_tiled,
            _ensure_asprs_classification_capacity,
            _resolve_output_path,
            _strip_copc_vlrs,
            _write_las,
        )

        config_path = str(ModelManager.get_config_path())
        model_path = str(ModelManager.get_model_path())

        # The class mapping is internal-only: it tells the inference
        # post-processor how to translate the model's 1-7 IDs into the
        # standard ASPRS codes.
        class_mapping = DEFAULT_CLASS_MAPPING

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
                    classifier_fn=filterPoints,
                    config_path=config_path,
                    model_path=model_path,
                    use_cuda=use_cuda,
                    class_mapping=class_mapping,
                    laspy_module=laspy,
                    progress_callback=stream_progress,
                    cancel_callback=feedback.isCanceled,
                    field_name=field_name,
                    tile_auto=tile_auto,
                    tile_size_m=None if tile_auto else tile_size_m,
                    buffer_m=tile_buffer_m,
                    info_callback=feedback.pushInfo,
                    warning_callback=feedback.pushWarning,
                )
            except InterruptedError:
                raise QgsProcessingException(self.tr("Cancelled."))

            if written is None:
                raise QgsProcessingException(
                    self.tr("Streaming classification returned no result.")
                )

            log_info(f"Streaming processing wrote {written}")
            if load_as_layer:
                self._add_output_layer(written, context, feedback)
            return {self.OUTPUT_FOLDER: str(output_folder)}

        feedback.pushInfo(self.tr(f"Loading {input_path.name}..."))
        with laspy.open(str(input_path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            las = reader.read()

        if len(las.x) == 0:
            raise QgsProcessingException(
                self.tr("Input file has zero points - nothing to classify.")
            )

        pcd = np.transpose(np.array([las.x, las.y, las.z]))

        def progress_cb(p):
            if feedback.isCanceled():
                raise InterruptedError()
            feedback.setProgress(min(100, max(0, int(p))))

        feedback.pushInfo(
            self.tr(
                f"Running classification on {'GPU' if use_cuda else 'CPU'}..."
            )
        )
        try:
            if tile_enabled:
                feedback.pushInfo(self.tr(
                    "Tiling enabled "
                    f"(auto={tile_auto}, tile={tile_size_m or 'auto'} m, "
                    f"buffer={tile_buffer_m:.0f} m)."
                ))
                preds = _classify_tiled(
                    pcd, filterPoints, config_path, model_path,
                    use_cuda, progress_cb, feedback.isCanceled,
                    auto=tile_auto,
                    tile_size_m=None if tile_auto else tile_size_m,
                    buffer_m=tile_buffer_m,
                )
            else:
                preds = filterPoints(
                    config_path, pcd, model_path,
                    if_bottom_only=False, use_efficient=True,
                    use_cuda=use_cuda, progress_callback=progress_cb,
                )
        except InterruptedError:
            raise QgsProcessingException(self.tr("Cancelled."))
        except Exception as exc:  # pragma: no cover - surfaced to user
            log_error(f"Inference failed: {exc}")
            raise QgsProcessingException(
                self.tr(f"Classification failed: {exc}")
            )

        if preds is None:
            raise QgsProcessingException(
                self.tr("Classifier returned no results.")
            )

        asprs = np.zeros_like(preds, dtype=np.int32)
        for mid, info in class_mapping.items():
            asprs[preds == mid] = info.asprs_code

        mapped_ids = list(class_mapping.keys())
        unmapped_mask = ~np.isin(preds, mapped_ids)
        unmapped_count = int(unmapped_mask.sum())
        if unmapped_count:
            missing_ids = sorted(np.unique(preds[unmapped_mask]).tolist())
            feedback.pushWarning(self.tr(
                f"{unmapped_count:,} prediction(s) had model IDs not in "
                f"the class mapping (IDs {missing_ids}); set to ASPRS 0."
            ))

        is_asprs_field = field_name.lower() == ASPRS_CLASSIFICATION_FIELD
        if is_asprs_field:
            try:
                max_code = int(asprs.max())
            except ValueError:
                max_code = 0
            las = _ensure_asprs_classification_capacity(las, laspy, max_code)
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
                setattr(las, field_name, asprs)
            else:
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name=field_name, type="int32",
                    description="AI Classification (3D SegFormer / TreeAIBox)",
                ))
                setattr(las, field_name, asprs)
            feedback.pushInfo(
                self.tr(
                    f"Writing to extra-byte field '{field_name}' "
                    "(non-standard, not ASPRS-compliant)."
                )
            )

        _strip_copc_vlrs(las)

        output_path = _resolve_output_path(input_path, output_folder, suffix)
        self._guard_input_output_collision(input_path, output_path)
        if output_path.exists():
            feedback.pushWarning(self.tr(
                f"Output {output_path.name} already exists - overwriting."
            ))
        feedback.pushInfo(self.tr(f"Writing {output_path.name}..."))
        _write_las(las, output_path, laspy)

        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        log_info(f"Processing algorithm wrote {output_path}")

        if load_as_layer:
            self._add_output_layer(output_path, context, feedback)

        return {self.OUTPUT_FOLDER: str(output_folder)}

    # ------------------------------------------------------------------
    def _add_output_layer(self, output_path: Path, context, feedback) -> None:
        """Add the classified file to the QGIS project as a point-cloud layer."""
        try:
            from qgis.core import QgsPointCloudLayer, QgsProject
        except ImportError:
            feedback.pushWarning(self.tr(
                "QgsPointCloudLayer not available in this QGIS version - "
                "cannot auto-load the result."
            ))
            return

        layer = QgsPointCloudLayer(str(output_path), output_path.stem, "pdal")
        if not layer.isValid():
            feedback.pushWarning(self.tr(
                f"Output written but could not be loaded as point-cloud "
                f"layer: {output_path}"
            ))
            return

        # Add to project (and to context so the Processing post-processor
        # also tracks the layer when the algorithm is run from a model).
        QgsProject.instance().addMapLayer(layer)
        feedback.pushInfo(
            self.tr(f"Loaded layer '{output_path.stem}' into the project.")
        )
