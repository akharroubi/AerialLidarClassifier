# Model provenance and licences

The plugin's original source is GPL-3.0-or-later; see LICENSE. Model weights
are separate downloads and are not included in the plugin ZIP.

## LitePT

Architecture and vendored implementation: Photogrammetry and Remote Sensing
Lab, ETH Zurich, https://github.com/prs-eth/LitePT. The upstream MIT notice is
retained in `core/litept/LICENSE.upstream`.

The released DALES 10 cm weights were trained by Abderrazzaq Kharroubi /
GeoScITY Lab, University of Liege. The maintainer confirmed ownership of this
trained model on 25 September 2026. The release retains the maintainer's
existing CC BY-NC 4.0 designation for these weights. This designation is not
an assertion that the training dataset automatically determines the licence
of a trained model. DALES is credited separately as training data.

Weight file: `litept_l_dales_10cm_ema_fp16.pth`

SHA-256: `849ba5089e629785fd64f5166cc35f999b758c68754573bf18122a277b09592b`

## SegFormer 3D / UrbanFiltering

Model and reference implementation: Zhouxin Xi, Natural Resources Canada,
TreeAIBox, https://github.com/NRCan/TreeAIBox. Crown Copyright, Government of
Canada. Existing upstream attribution is retained. Downloaded weights remain
unchanged and are designated CC BY-NC 4.0 in the model registry.

On 25 September 2026, the plugin maintainer confirmed that the authors had
permitted SegFormer's use in this plugin, which was already publicly available
on the QGIS repository. This records the maintainer's confirmation; it does
not reproduce a permission document, assert a broader grant, or change the
upstream model licence. The maintainer is the custodian of the permission
record. QGIS publication alone is not evidence of a licence grant.

Weight file: `urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth`

SHA-256: `cddb791041d46a2e7be3c53d6e94157c9af2218e7ea5595bf3134c4390c1fbc0`

## Runtime dependencies

Python, PyTorch, torchvision, laspy, lazrs, scipy, NumPy, timm, numpy_indexed,
spconv and their dependencies retain their own licences. They are installed
separately into a per-user environment; they are not relicensed by this file.
