"""v1.3 integration checks of the Railway model with real QGIS/Qt and LAS I/O.

Run with QGIS's python-qgis-ltr.bat (QGIS 3) or python-qgis.bat (QGIS 4).
Inference is stubbed (real CUDA inference: test_litept_rail_backend.py and
qgis_rail_gpu.py). No network, GPU, installed plugin or user settings needed.
Shares the synthetic source file and Processing patches of qgis_mls_checks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Importing the MLS checks starts QGIS (offscreen) before anything imports SSL.
import qgis_mls_checks as base  # noqa: E402

import json  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from unittest.mock import patch  # noqa: E402

import laspy  # noqa: E402
import numpy as np  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsPointCloudLayer, QgsProcessingContext, QgsProcessingException, QgsProcessingFeedback,
)
from qgis.PyQt.QtCore import QTimer  # noqa: E402

app = base.app
module = base.module
registry = base.registry
tasks = base.tasks
mapping = base.mapping
processing = base.processing
manager_module = base.manager_module
config = base.config
RAIL = registry.get_model('litept_l_rail_5cm')
MLS = base.MLS
DEFAULT_CODES = [2, 10, 15, 14, 5, 64, 1]


def rail_predictions(xyz, progress=None, cancel=None):
    if progress is not None:
        progress(80.)
    return ((np.rint(xyz[:, 0]).astype(np.int64) - 500000) % 7 + 1).astype(np.int32)


class RecordingFeedback(QgsProcessingFeedback):
    def __init__(self):
        super().__init__()
        self.messages = []

    def pushInfo(self, info):  # noqa: N802
        self.messages.append(info)
        super().pushInfo(info)


class RailStubBackend(base.StubBackend):
    def predict(self, xyz, progress=None, cancel=None):
        return rail_predictions(xyz, progress, cancel)


def expected_codes(overrides=None):
    codes = list(DEFAULT_CODES)
    for class_id, code in (overrides or {}).items():
        codes[class_id - 1] = code
    # make_source writes 27 points with x = 0..26 m.
    return np.array([codes[i % 7] for i in range(27)], dtype=np.uint8)


class RailChecks(unittest.TestCase):
    assert_preserved = base.MlsChecks.assert_preserved

    def assert_metadata(self, cloud, field, codes):
        records = [v for v in cloud.vlrs if v.user_id == 'AerialLiDAR' and v.record_id == 1]
        self.assertEqual(len(records), 1)
        data = json.loads(records[0].record_data_bytes().decode('utf8'))
        self.assertEqual(data['model'], RAIL.id)
        self.assertEqual(data['weights_sha256'], RAIL.weights_sha256)
        self.assertEqual(data['field'], field)
        self.assertEqual(data['encoding'], 'ASPRS')
        self.assertEqual(sorted(data['classes']), [str(i) for i in range(1, 8)])
        for class_id, code in codes.items():
            self.assertEqual(data['classes'][str(class_id)]['asprs'], code)
        self.assertEqual(data['classes']['2']['name'], 'Rail')

    def test_registry_order_processing_enum_and_flags(self):
        # The legacy MODEL parameter stores an engine index: the first three never move.
        self.assertEqual([s.id for s in registry.MODELS],
                         ['litept_l_dales_10cm', 'segformer3d_urbanfiltering', MLS.id, RAIL.id])
        self.assertEqual([c.id for c in registry.CLASSIFIERS], ['airborne', 'mobile_mapping', 'railway'])
        self.assertEqual([c.display_name for c in registry.CLASSIFIERS],
                         ['Airborne classifier', 'Mobile mapping classifier', 'Railway classifier'])
        self.assertEqual(registry.AIRBORNE.engines,
                         (registry.LITEPT_L_DALES, registry.SEGFORMER3D_URBANFILTERING))
        self.assertEqual(registry.RAILWAY.engines, (RAIL,))
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        self.assertEqual(algorithm.name(), 'classify_point_cloud')
        classifier = algorithm.parameterDefinition('CLASSIFIER')
        self.assertEqual(classifier.options(), [c.display_name for c in registry.CLASSIFIERS])
        self.assertEqual(classifier.defaultValue(), 0)
        self.assertIsNone(algorithm.parameterDefinition('MODEL'))
        self.assertFalse(algorithm.flags() & processing._ALG_DEPRECATED)
        self.assertEqual([mapping.supports_output_codes(s) for s in registry.MODELS],
                         [False, False, True, True])
        self.assertIn('Railway classifier', algorithm.shortHelpString())
        self.assertIn('Rail 10', algorithm.shortHelpString())
        self.assertIn('SegFormer 3D engine', algorithm.shortHelpString())
        self.assertIn('aeriallidar:classify_lidar', algorithm.shortHelpString())
        # The 1.0-1.2 algorithm keeps its id and its visible MODEL parameter,
        # flagged deprecated so the Toolbox lists only the classifier one.
        legacy_alg = processing.ClassifyLidarAlgorithm()
        legacy_alg.initAlgorithm()
        self.assertEqual(legacy_alg.name(), 'classify_lidar')
        self.assertTrue(legacy_alg.flags() & processing._ALG_DEPRECATED)
        self.assertIsNone(legacy_alg.parameterDefinition('CLASSIFIER'))
        legacy = legacy_alg.parameterDefinition('MODEL')
        self.assertEqual(legacy.options(), [s.display_name for s in registry.MODELS])
        self.assertEqual(legacy.defaultValue(), 0)
        self.assertFalse(legacy.flags() & processing._FLAG_ADVANCED)
        self.assertIn('3 LitePT-L Railway', legacy_alg.shortHelpString())
        self.assertIn('aeriallidar:classify_point_cloud', legacy_alg.shortHelpString())
        # Both share every other parameter, in the same order.
        names = [d.name() for d in algorithm.parameterDefinitions()]
        legacy_names = [d.name() for d in legacy_alg.parameterDefinitions()]
        self.assertEqual(names[0], 'INPUT')
        self.assertEqual(names[1:2] + legacy_names[1:2], ['CLASSIFIER', 'MODEL'])
        self.assertEqual(names[2:], legacy_names[2:])
        provider = module('processing.provider').AerialLidarProvider()
        from qgis.core import QgsApplication
        registry_ = QgsApplication.processingRegistry()
        if registry_.providerById(provider.id()) is None:
            registry_.addProvider(provider)
        self.assertIsNotNone(registry_.algorithmById('aeriallidar:classify_point_cloud'))
        self.assertIsNotNone(registry_.algorithmById('aeriallidar:classify_lidar'))

    def test_default_codes_are_asprs_where_a_class_exists(self):
        codes = {i: c.asprs_code for i, c in RAIL.class_mapping.items()}
        self.assertEqual(codes, {1: 2, 2: 10, 3: 15, 4: 14, 5: 5, 6: 64, 7: 1})
        meanings = [mapping.asprs_code_name(c) for c in codes.values()]
        self.assertEqual(meanings, ['Ground', 'Rail', 'Transmission tower', 'Wire conductor (phase)',
                                    'High vegetation', 'User definable', 'Unclassified'])
        # The only user-defined default is the lowest user-definable code.
        self.assertEqual([c for c in codes.values() if c >= 64], [64])

    def test_task_modes_default_and_custom_codes_preserve_attributes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = base.make_source(source)
            source_bytes = source.read_bytes()
            for overrides in (None, {2: 200, 6: 17, 7: 0}):
                for mode in ('memory', 'tiled', 'streaming'):
                    with self.subTest(mode=mode, overrides=overrides):
                        output = folder / f"{mode}_{'custom' if overrides else 'default'}.laz"
                        task = tasks.ClassificationTask(
                            [source], folder, '_out', RAIL.id, 'cuda', 'classification',
                            tile_enabled=mode != 'memory', tile_auto=False, tile_size_m=10,
                            tile_buffer_m=1, tile_streaming=mode == 'streaming',
                            units_override='metre', output_codes=overrides)
                        task._process_one(source, output, rail_predictions, laspy, 0, 100)
                        result = laspy.read(output)
                        # 64 and 200 exceed the 5-bit legacy range: LAS 1.4 upgrade.
                        self.assertEqual(result.point_format.id, 7)
                        self.assertEqual(str(result.header.version), '1.4')
                        np.testing.assert_array_equal(result.classification, expected_codes(overrides))
                        self.assertEqual(list(result.point_format.extra_dimension_names), ['precise'])
                        self.assert_preserved(original, result)
                        written = {i + 1: c for i, c in enumerate(DEFAULT_CODES)}
                        written.update(overrides or {})
                        self.assert_metadata(result, 'classification', written)
                        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual(RAIL.class_mapping[6].asprs_code, 64)

    def test_legacy_file_stays_legacy_when_codes_fit_five_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = base.make_source(source, extras=False)
            output = folder / 'small_codes.las'
            task = tasks.ClassificationTask([source], folder, '_out', RAIL.id, 'cuda',
                                            'classification', units_override='metre',
                                            output_codes={6: 20})
            task._process_one(source, output, rail_predictions, laspy, 0, 100)
            result = laspy.read(output)
            self.assertEqual(result.point_format.id, 3)
            np.testing.assert_array_equal(result.classification, expected_codes({6: 20}))
            self.assert_preserved(original, result)

    def test_rail_refuses_extra_class_field_without_writing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            base.make_source(source)
            original_bytes = source.read_bytes()
            for stream in (False, True):
                with self.assertRaisesRegex(ValueError, 'LitePT-L Railway.*classification'):
                    tasks.ClassificationTask([source], folder, '_raw', RAIL.id, 'cuda', 'rail_ids',
                        tile_enabled=stream, tile_auto=False, tile_size_m=10, tile_buffer_m=0,
                        tile_streaming=stream, units_override='metre')
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(list(folder.iterdir()), [source])

    def test_outputs_open_in_qgis_without_new_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = base.make_source(source, extras=False)
            for stream in (False, True):
                for extension in ('.las', '.laz'):
                    with self.subTest(stream=stream, extension=extension):
                        output = folder / (('stream' if stream else 'memory') + extension)
                        task = tasks.ClassificationTask([source], folder, '_out', RAIL.id, 'cuda',
                            'classification', tile_enabled=stream, tile_auto=False, tile_size_m=10,
                            tile_buffer_m=0, tile_streaming=stream, units_override='metre')
                        task._process_one(source, output, rail_predictions, laspy, 0, 100)
                        written = laspy.read(output)
                        self.assertEqual(list(written.point_format.extra_dimension_names), [])
                        self.assert_preserved(original, written)
                        options = QgsPointCloudLayer.LayerOptions()
                        options.skipIndexGeneration = True
                        options.skipStatisticsCalculation = True
                        options.skipCrsValidation = True
                        layer = QgsPointCloudLayer(str(output), output.stem, 'pdal', options)
                        self.assertTrue(layer.isValid(), layer.error().message())
                        self.assertEqual(layer.dataProvider().pointCount(), 27)

    def test_classified_style_lists_every_rail_class(self):
        helpers = module('utils.helpers')
        categories = helpers._point_cloud_categories(RAIL.class_mapping)
        self.assertEqual([c.value() for c in categories], [1, 2, 5, 10, 14, 15, 64])
        labels = {c.value(): c.label() for c in categories}
        self.assertEqual(labels[10], 'Rail')
        self.assertEqual(labels[64], 'Fence')
        merged = mapping.output_model_spec(RAIL, {3: 14})
        labels = {c.value(): c.label() for c in helpers._point_cloud_categories(merged.class_mapping)}
        self.assertEqual(labels[14], 'Pole / Wire')

    def processing_params(self, algorithm, source, folder, stream=False, codes='{"2":200,"6":17}'):
        params = base.MlsChecks.processing_params(self, algorithm, source, folder, stream)
        del params['MODEL']
        params['CLASSIFIER'] = 2
        params[algorithm.OUTPUT_CODES_JSON] = codes
        return params

    def test_processing_railway_classifier_parses_codes_and_writes_classification(self):
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        for stream, codes, overrides in ((False, '{"2":200,"6":17}', {2: 200, 6: 17}),
                                         (True, '', None)):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as folder, \
                    ExitStack() as stack:
                folder = Path(folder)
                source = folder / 'input.las'
                original = base.make_source(source)
                backend = RailStubBackend()
                factory = base.MlsChecks.processing_patches(self, stack, backend)
                feedback = RecordingFeedback()
                result = algorithm.processAlgorithm(
                    self.processing_params(algorithm, source, folder, stream, codes),
                    QgsProcessingContext(), feedback)
                written = laspy.read(result[algorithm.OUTPUT_FILE])
                self.assertEqual(factory.call_args.args[0].id, RAIL.id)
                self.assertEqual(backend.loaded, ['cuda'])
                self.assertEqual(backend.unloaded, 1)
                np.testing.assert_array_equal(written.classification, expected_codes(overrides))
                self.assert_preserved(original, written)
                expected = {i + 1: c for i, c in enumerate(DEFAULT_CODES)}
                expected.update(overrides or {})
                self.assert_metadata(written, 'classification', expected)
                self.assertTrue(any(m.startswith('LitePT-L Railway output codes:')
                                    for m in feedback.messages), feedback.messages)

    def test_processing_rejects_bad_codes_field_and_cpu_before_inference(self):
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            source = Path(folder) / 'input.las'
            base.make_source(source)
            factory = base.MlsChecks.processing_patches(self, stack, RailStubBackend())
            parameters = self.processing_params(algorithm, source, folder)
            # ID 8 exists for Mobile Mapping but not for Railway.
            for invalid in ('{"8":3}', '{"9":1}', '{"6":256}', '{"0":3}', '{"6":true}', '{broken}'):
                with self.subTest(value=invalid), self.assertRaises(QgsProcessingException):
                    parameters[algorithm.OUTPUT_CODES_JSON] = invalid
                    algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            parameters[algorithm.OUTPUT_CODES_JSON] = ''
            parameters[algorithm.FIELD_NAME] = 'rail_extra'
            with self.assertRaisesRegex(QgsProcessingException, 'classification'):
                algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            factory.assert_not_called()
            self.assertFalse((Path(folder) / 'input_processed.las').exists())
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            source = Path(folder) / 'input.las'
            base.make_source(source)
            factory = base.MlsChecks.processing_patches(self, stack, RailStubBackend(), cuda=False)
            parameters = self.processing_params(algorithm, source, folder)
            parameters[algorithm.DEVICE] = 0
            with self.assertRaisesRegex(QgsProcessingException, 'Railway'):
                algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            factory.assert_not_called()

    def run_processing(self, parameters, cuda=True, litept_ready=True, legacy=False, context=None):
        """Processing run with stubbed inference; returns (engine id, warnings, infos)."""
        algorithm = (processing.ClassifyLidarAlgorithm() if legacy
                     else processing.ClassifyPointCloudAlgorithm())
        algorithm.initAlgorithm()
        readiness = module('utils.backend_readiness')
        with ExitStack() as stack:
            backend = RailStubBackend()
            factory = base.MlsChecks.processing_patches(self, stack, backend, cuda=cuda)
            stack.enter_context(patch.object(readiness, 'litept_dependency_status',
                                             return_value=(litept_ready, '' if litept_ready else 'spconv missing')))
            warnings = []

            class Feedback(RecordingFeedback):
                def pushWarning(self, warning):  # noqa: N802
                    warnings.append(warning)
                    super().pushWarning(warning)

            feedback = Feedback()
            algorithm.processAlgorithm(parameters, context or QgsProcessingContext(), feedback)
            return factory.call_args.args[0].id, warnings, feedback.messages

    def test_processing_airborne_engine_selection_and_legacy_model_indexes(self):
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.las'
            base.make_source(source)
            base_params = self.processing_params(algorithm, source, folder, codes='')
            airborne = dict(base_params, **{'CLASSIFIER': 0, algorithm.FIELD_NAME: 'classification'})
            # CUDA and LitePT ready: LitePT-L Airborne, nothing to explain.
            engine, warnings, _ = self.run_processing(airborne)
            self.assertEqual(engine, 'litept_l_dales_10cm')
            self.assertFalse([w for w in warnings if 'engine' in w], warnings)
            # Dependencies missing on a CUDA machine: SegFormer 3D, with the reason.
            engine, warnings, _ = self.run_processing(airborne, litept_ready=False)
            self.assertEqual(engine, 'segformer3d_urbanfiltering')
            self.assertTrue(any('spconv missing' in w and 'SegFormer 3D' in w for w in warnings), warnings)
            # Explicit CPU device: SegFormer 3D on the CPU.
            engine, warnings, _ = self.run_processing(dict(airborne, **{algorithm.DEVICE: 2}))
            self.assertEqual(engine, 'segformer3d_urbanfiltering')
            self.assertTrue(any("does not run on 'cpu'" in w for w in warnings), warnings)
            # Auto device without CUDA: SegFormer 3D.
            engine, _, _ = self.run_processing(dict(airborne, **{algorithm.DEVICE: 0}), cuda=False)
            self.assertEqual(engine, 'segformer3d_urbanfiltering')
            # Railway and Mobile mapping never fall back to an airborne engine.
            for index, expected in ((1, MLS.id), (2, RAIL.id), ('2', RAIL.id), (2.0, RAIL.id)):
                engine, _, _ = self.run_processing(dict(airborne, **{'CLASSIFIER': index}))
                self.assertEqual(engine, expected)
            with self.assertRaisesRegex(QgsProcessingException, 'Railway'):
                self.run_processing(dict(airborne, **{'CLASSIFIER': 2, algorithm.DEVICE: 2}))
            # Without CLASSIFIER (Python calls that omit it) the default is Airborne.
            engine, _, _ = self.run_processing({k: v for k, v in airborne.items() if k != 'CLASSIFIER'})
            self.assertEqual(engine, 'litept_l_dales_10cm')
            # 1.0 - 1.2 algorithm: MODEL selects that exact engine, never another.
            legacy = {k: v for k, v in airborne.items() if k != 'CLASSIFIER'}
            for index, expected in ((0, 'litept_l_dales_10cm'), (1, 'segformer3d_urbanfiltering'),
                                    (2, MLS.id), (3, RAIL.id), ('2', MLS.id), (None, 'litept_l_dales_10cm')):
                with self.subTest(model=index):
                    params = dict(legacy, MODEL=index) if index is not None else dict(legacy)
                    engine, warnings, _ = self.run_processing(params, legacy=True)
                    self.assertEqual(engine, expected)
                    self.assertFalse([w for w in warnings if 'engine' in w], warnings)
            # MODEL=0 keeps its 1.2 meaning even without CUDA: LitePT-L, refused on the CPU.
            with self.assertRaisesRegex(QgsProcessingException, "does not run on 'cpu'"):
                self.run_processing(dict(legacy, MODEL=0, **{algorithm.DEVICE: 2}), legacy=True)

    def test_out_of_range_or_invalid_choices_are_refused_also_from_expressions(self):
        """An index no version ever had is refused, never mapped to the default."""
        from qgis.core import QgsProperty
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.las'
            base.make_source(source)
            params = dict(self.processing_params(algorithm, source, folder, codes=''),
                          **{algorithm.FIELD_NAME: 'classification'})
            legacy = {k: v for k, v in params.items() if k != 'CLASSIFIER'}
            bad = (3, '9', -2, '2.9', 2.5, 'garbage', True, QgsProperty.fromExpression('99'),
                   QgsProperty.fromExpression("'text'"))
            for value in bad:
                with self.subTest(classifier=value), \
                        self.assertRaisesRegex(QgsProcessingException, 'CLASSIFIER must be an integer from 0 to 2'):
                    self.run_processing(dict(params, CLASSIFIER=value))
            for value in (4, '7', -1, QgsProperty.fromExpression('4'), QgsProperty.fromExpression('1 + 3')):
                with self.subTest(model=value), \
                        self.assertRaisesRegex(QgsProcessingException, 'MODEL must be an integer from 0 to 3'):
                    self.run_processing(dict(legacy, MODEL=value), legacy=True)
            # Valid expressions select what they compute.
            engine, _, _ = self.run_processing(dict(params, CLASSIFIER=QgsProperty.fromExpression('1 + 1')))
            self.assertEqual(engine, RAIL.id)
            engine, _, _ = self.run_processing(dict(legacy, MODEL=QgsProperty.fromExpression('3')),
                                               legacy=True)
            self.assertEqual(engine, RAIL.id)

    def test_graphical_modeler_editor_keeps_the_chosen_model(self):
        """Opening and accepting a saved model step keeps its model or classifier.

        Uses QGIS's own ModelerParametersDialog, as a user editing a step does.
        """
        from qgis.core import (QgsApplication, QgsProcessingModelAlgorithm,
                               QgsProcessingModelChildAlgorithm,
                               QgsProcessingModelChildParameterSource)
        plugins_dir = Path(QgsApplication.pkgDataPath()) / 'python' / 'plugins'
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        from processing.core.Processing import Processing
        Processing.initialize()
        from processing.modeler.ModelerParametersDialog import ModelerParametersDialog
        provider = module('processing.provider').AerialLidarProvider()
        registry_ = QgsApplication.processingRegistry()
        if registry_.providerById(provider.id()) is None:
            registry_.addProvider(provider)
        cases = [('aeriallidar:classify_lidar', 'MODEL', index) for index in range(4)]
        cases += [('aeriallidar:classify_point_cloud', 'CLASSIFIER', index) for index in range(3)]
        for algorithm_id, name, index in cases:
            with self.subTest(algorithm=algorithm_id, value=index):
                algorithm = registry_.algorithmById(algorithm_id)
                model = QgsProcessingModelAlgorithm()
                child = QgsProcessingModelChildAlgorithm(algorithm.id())
                child.setChildId('step')
                for key, value in {name: index, 'INPUT': 'input.las', 'OUTPUT_FOLDER': '.',
                                   'DEVICE': 0, 'FIELD_NAME': 'classification'}.items():
                    child.addParameterSources(key, [QgsProcessingModelChildParameterSource.fromStaticValue(value)])
                model.addChildAlgorithm(child)
                dialog = ModelerParametersDialog(algorithm, model, 'step')
                try:
                    edited = dialog.widget.widget.createAlgorithm()
                finally:
                    dialog.close()
                values = {key: sources[0].staticValue()
                          for key, sources in edited.parameterSources().items() if sources}
                self.assertEqual(int(values[name]), index, values)
                other = 'CLASSIFIER' if name == 'MODEL' else 'MODEL'
                self.assertNotIn(other, values)
                resolved = algorithm._requested_engines(values, QgsProcessingContext(), QgsProcessingFeedback())
                expected = (registry.MODELS[index],) if name == 'MODEL' else registry.CLASSIFIERS[index].engines
                self.assertEqual(resolved[1], expected)

    def test_rail_weights_are_published_not_bundled(self):
        self.assertEqual(RAIL.weights_urls, (
            'https://github.com/akharroubi/AerialLidarClassifier/releases/download/v1.3/'
            'litept_l_rail_5cm_fp16.pth',))
        card = json.loads(RAIL.config_path.read_text(encoding='utf-8'))
        self.assertEqual(card['weights_file'], RAIL.weights_filename)
        self.assertAlmostEqual(card['weights_bytes'] / 1e6, RAIL.weights_size_mb, delta=0.1)
        self.assertNotIn('weights_sha256', card)

    def test_dock_editor_persistence_and_field_name_across_models(self):
        panel_module = module('gui.main_panel')
        dialog_module = module('dialogs.class_mapping_dialog')
        prefix = config.SETTINGS_PREFIX
        settings = {f'{prefix}/settings_version': 2,
                    f'{prefix}/model_id': 'litept_l_dales_10cm',
                    f'{prefix}/field_name': 'my_raw_ids',
                    f'{prefix}/airborne_field_name': 'my_raw_ids',
                    f'{prefix}/output_codes/{MLS.id}': json.dumps({'5': 99})}

        class Settings:
            def value(self, key, default=None, **kwargs):
                return settings.get(key, default)

            def setValue(self, key, value):
                settings[key] = value

        with patch.object(panel_module, 'QgsSettings', Settings), \
             patch.object(panel_module, 'get_gpu_info', return_value={'available': False}), \
             patch.object(manager_module.ModelManager, 'is_model_available', return_value=True):
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(panel.field_edit.text(), 'my_raw_ids')
                self.assertTrue(panel.mms_codes_row.isHidden())
                self.assertEqual([panel.model_combo.itemText(i) for i in range(panel.model_combo.count())],
                                 ['Airborne classifier', 'Mobile mapping classifier', 'Railway classifier'])
                # 1.2 saved the LitePT-L Airborne model id: Airborne classifier, which
                # runs its SegFormer 3D engine on this GPU-less test machine.
                self.assertEqual(panel.model_combo.currentData(), 'airborne')
                self.assertEqual(panel._current_spec(), registry.SEGFORMER3D_URBANFILTERING)
                self.assertTrue(panel.model_gate_ok)
                self.assertIn('SegFormer 3D engine', panel.model_label.text())
                # Airborne -> Mobile mapping -> Railway -> Airborne keeps the
                # user's raw-ID field name (1.2 replaced it by 'classification').
                for index, (classifier_id, spec) in enumerate(
                        (('mobile_mapping', MLS), ('railway', RAIL)), start=1):
                    panel.model_combo.setCurrentIndex(index)
                    self.assertEqual(panel.model_combo.currentData(), classifier_id)
                    self.assertEqual(panel._current_spec(), spec)
                    self.assertFalse(panel.mms_codes_row.isHidden())
                    self.assertEqual(panel.field_edit.text(), 'classification')
                    self.assertFalse(panel.field_edit.isEnabled())
                self.assertIn('Rail: 10', panel.mms_codes_btn.toolTip())
                self.assertIn('Fence: 64', panel.mms_codes_btn.toolTip())
                self.assertFalse(panel.model_gate_ok)
                self.assertIn('LitePT-L Railway', panel.model_gate_message)
                editor_error = []

                def edit_and_accept():
                    dialog = app.activeModalWidget()
                    try:
                        self.assertIsInstance(dialog, dialog_module.ClassMappingDialog)
                        self.assertEqual(dialog.windowTitle(), 'LitePT-L Railway output class codes')
                        self.assertEqual(len(dialog._spins), 7)
                        self.assertEqual(dialog.table.item(1, 1).text(), 'Rail')
                        self.assertEqual(dialog.table.item(1, 4).text(), 'Rail')
                        self.assertEqual(dialog.table.item(5, 4).text(), 'User definable')
                        dialog._spins[2].setValue(30)
                        self.assertEqual(dialog.table.item(1, 4).text(), 'Reserved')
                        dialog._spins[2].setValue(10)
                        dialog._spins[6].setValue(65)
                        dialog.accept()
                    except BaseException as error:
                        editor_error.append(error)
                        if dialog is not None:
                            dialog.reject()

                QTimer.singleShot(0, edit_and_accept)
                panel._edit_output_codes()
                if editor_error:
                    raise editor_error[0]
                self.assertEqual(panel._output_codes_by_model[RAIL.id][6], 65)
                self.assertEqual(panel._output_codes_by_model[MLS.id][5], 99)
                # Class lists: every selector item, the selector and the status line.
                from qgis.PyQt.QtCore import Qt
                tips = [panel.model_combo.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(3)]
                self.assertIn('On this computer: SegFormer 3D Airborne (30 cm), 7 classes', tips[0])
                self.assertIn('Building</td><td align=\'right\'>&nbsp;&nbsp;6<', tips[0])
                self.assertIn('Pole like</td><td align=\'right\'>&nbsp;&nbsp;99<', tips[1])
                self.assertIn('LitePT-L Railway (5 cm), 7 classes', tips[2])
                self.assertIn("Rail</td><td align='right'>&nbsp;&nbsp;10</td><td>&nbsp;&nbsp;Rail<", tips[2])
                # The saved custom fence code (65) shows at once, with its meaning.
                self.assertIn("Fence</td><td align='right'>&nbsp;&nbsp;65</td><td>&nbsp;&nbsp;User definable", tips[2])
                self.assertIn('#E31A1C', tips[2])
                self.assertEqual(panel.model_combo.toolTip(), tips[2])
                self.assertTrue(panel.model_label.toolTip().endswith(tips[2]))
                self.assertTrue(panel.model_label.toolTip().startswith(
                    'LitePT-L Railway (5 cm) runs on NVIDIA CUDA GPUs only.<hr>'))
                saved = json.loads(settings[f'{prefix}/output_codes/{RAIL.id}'])
                self.assertEqual(saved, {'1': 2, '2': 10, '3': 15, '4': 14, '5': 5, '6': 65, '7': 1})
                panel.model_combo.setCurrentIndex(0)
                self.assertTrue(panel.mms_codes_row.isHidden())
                self.assertTrue(panel.field_edit.isEnabled())
                self.assertEqual(panel.field_edit.text(), 'my_raw_ids')
                panel.model_combo.setCurrentIndex(2)
                panel._save_settings()
                self.assertEqual(settings[f'{prefix}/classifier_id'], 'railway')
            finally:
                panel.close()
            restored = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(restored.model_combo.currentData(), 'railway')
                self.assertEqual(restored._output_codes_by_model[RAIL.id][6], 65)
                self.assertEqual(restored._output_codes_by_model[MLS.id][5], 99)
                dialog = dialog_module.ClassMappingDialog(RAIL, restored._output_codes_by_model[RAIL.id])
                capture_dir = base.os.environ.get('ALC_UI_CAPTURE_DIR')
                if capture_dir:
                    capture_dir = Path(capture_dir)
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    dialog.show()
                    app.processEvents()
                    self.assertTrue(dialog.grab().save(str(capture_dir / 'rail_output_codes.png')))
                self.assertEqual(dialog.output_codes()[6], 65)
                dialog._reset_defaults()
                self.assertEqual(dialog.output_codes(), dict(enumerate(DEFAULT_CODES, start=1)))
                self.assertEqual(dialog.table.item(5, 4).text(), 'User definable')
                dialog.close()
            finally:
                restored.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
