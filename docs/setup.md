# Setup and data contracts

## Source-only sample

The tracking module uses the Python standard library. Run from the repository root:

```bash
python -m shelf_audit.track_video examples/predictions.json --out outputs/sample_review --reconnect
```

This replays saved predictions, not the neural networks. Open the generated HTML locally. Input frames contain timestamps, dimensions, and detections with source-pixel `bbox_xyxy`, detector score, and recognizer predictions. The included example illustrates the full schema.

## Full environment

The documented target versions are Python 3.10.8, PyTorch 1.13.0, Torchvision 0.14.0, OpenCV 4.6.0, and CUDA 11.7. These targets are untested with the current code. `requirements.txt` remains unchanged and pins the newer implementation environment. Install in a virtual environment from the repository root. No package installation step is needed beyond dependencies: modules run with `python -m shelf_audit.…`.

The full app expects:

```text
data/catalog_crops/manifest.json
data/catalog_crops/<product>/<image>.jpg
artifacts/catalog_index/index.json
artifacts/catalog_index/embeddings.npy
artifacts/product_detector/corrected.pt
```

None of these private working artifacts are in the source repository. The catalog manifest is a list of records with `class_id`, `product`, and `crop_file` relative to the catalog directory. Indexing adds image hashes and preprocessing metadata. Rebuild the index after moving a catalog to another machine; existing index metadata can contain an absolute catalog root.

```powershell
.venv/Scripts/python.exe -m shelf_audit index
.venv/Scripts/python.exe -m shelf_audit.audit_app
```

The app runs on port 8766. It needs a detector checkpoint; indexing only creates the recognizer index. Saved-example links in the app refer to local evaluation artifacts and are not populated by a source-only clone. Use the README sample instead.

## CLI pipeline

```powershell
.venv/Scripts/python.exe -m shelf_audit.process_video your_video.mov --checkpoint artifacts/product_detector/corrected.pt --out outputs/run/detection
.venv/Scripts/python.exe -m shelf_audit.refine_predictions outputs/run/detection/results.json outputs/run/refined.json
.venv/Scripts/python.exe -m shelf_audit.track_video outputs/run/refined.json --out outputs/run/tracking --reconnect
.venv/Scripts/python.exe -m shelf_audit.render_tracks your_video.mov outputs/run/tracking
```

Choose fresh output directories. Rendered video omits audio and uses nominal frame rate; source timestamps remain in JSON.

## Training the current research dataset

The current training loader is specific to the working dataset. It is not a generic COCO importer. It needs the original frame manifest, approved query annotations, upright frame JPEGs, and corrected frame JPEGs. The checked-in JSON specifications provide box coordinates but do not include source images.

```powershell
.venv/Scripts/python.exe -m shelf_audit.detector --extra-boxes shelf_audit/detector_extra_boxes.json --corrections shelf_audit/corrected_boxes.json --checkpoint artifacts/product_detector/new_candidate.pt
```

Corrected frame keys such as `v3_98` identify zero-based decoded frame 98 in rearranged video 3. Save upright source-resolution images to `data/corrected_detector/v3_98.jpg`; annotations use 540×960 preview coordinates scaled by two. The loader assumes 1080×1920 upright source frames. Adapt this contract before using a different dataset.

Model training and evaluation details: [corrected-model experiment](experiments/corrected_model_results.md). A fully portable dataset/weight release is future work; this repository does not claim one-command reproduction of unpublished assets.
