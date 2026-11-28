# AI tracker model

Trackwright's optional AI assist uses **AllTracker** (Adam W. Harley et al.,
*AllTracker: Efficient Dense Point Tracking at High Resolution*, ICCV 2025,
https://github.com/aharley/alltracker), MIT licence (see `ALLTRACKER_LICENSE.txt`
in the `ai-models` release).

The CI job `ai-model` builds the files the app downloads: it checks out the authors'
code at a fixed commit, applies `patch_alltracker.py` (export fixes only; results are
unchanged), downloads the authors' `alltracker.pth`, runs `export_onnx.py`, and
publishes `at_encoder.onnx`, `at_window.onnx`, `SHA256SUMS.txt` and the licence to
the `ai-models` release. The ONNX files match the PyTorch model to 0.0002 px.
