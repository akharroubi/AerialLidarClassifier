# Translations

Compiled translations (`*.qm`) are loaded automatically based on the QGIS
locale (`locale/userLocale` in QSettings). Place them in this directory
with the filename pattern `aerial_lidar_classifier_<locale>.qm`
(e.g. `aerial_lidar_classifier_fr.qm`).

To generate translation files:

1. Update the `.pro` file (or use `pylupdate5` / `pylupdate6`) to extract
   strings wrapped in `tr(...)` from the source.
2. Translate the resulting `.ts` files with Qt Linguist.
3. Compile with `lrelease` into `.qm` files in this folder.
