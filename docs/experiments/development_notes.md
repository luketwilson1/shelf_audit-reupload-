# ShelfAudit: video product-order review

A local baseline that identifies a cropped retail product by comparing it against a reference catalog. It uses a frozen pretrained ResNet-50 CNN with its classification head removed, producing normalized 2,048-dimensional embeddings. Adding catalog photos requires rebuilding the index, not retraining the CNN.

Current reference catalog: 56 user-approved crops, 12 product identities. The pipeline detects packages, recognizes crops, tracks repeated views, and proposes left-to-right order for a single shelf row. Unique inventory counts are not validated.

## Launch the interview demo

```powershell
.venv/Scripts/python.exe -m shelf_audit.audit_app
```

Open **http://127.0.0.1:8766**. Choose an MP4/MOV, click **Process video**, and follow the stage updates. The result includes tracked playback, product evidence crops, candidate order, and unresolved tracks. Three saved fresh-video examples are available without another inference run. Uploaded videos and outputs persist locally in `outputs/app_runs`; no uploads are sent to Drive and inference never retrains the model. One job runs at a time. Keep the server running while a job processes. In-memory job status does not survive a restart; completed artifacts remain on disk.

The app requires the local `artifacts/product_detector/corrected.pt` checkpoint and `artifacts/catalog_index` with its catalog images. Data and weights are gitignored and are **not included in a source-only clone**. Restore these local artifacts or prepare annotated data and follow the training commands in `CORRECTED_MODEL_RESULTS.md`; `index` alone does not build a detector. Install the pinned dependencies below first. The older crop-only demo remains available on port 8765.

## What to explain in an interview

1. **Detection:** COCO-pretrained Faster R-CNN with MobileNetV3 FPN, fine-tuned on individual package boxes. Corrected merged boxes reduced extra proposals on development frames.
2. **Recognition:** frozen ResNet-50 embeddings compared with a small product catalog; cosine similarity and margin trigger review rather than pretending to be calibrated confidence.
3. **Tracking:** spatial overlap and motion estimates connect observations. Short reconnections require supporting identity evidence. Offline identity voting uses later observations too.
4. **Order:** co-visible product positions become pairwise left/right constraints. Ambiguous or contradictory evidence remains flagged.
5. **Evaluation:** earlier rearranged footage was tested, reviewed, then explicitly promoted to development data. A frozen corrected model subsequently processed three freshly recorded videos without training on them.

The fresh tests (`IMG_1849`, `IMG_1850`, `IMG_1851`) each produced 12 identified tracks, 12 distinct product identities, zero persistent unknowns, and the same candidate order. The user gave general visual approval. Two clips retain conservative order-evidence flags, and brief tracks remain. This is **not** 100% detection accuracy or a validated count benchmark: the footage shares a household, physical packages, and one arrangement. Known limitations include merged boxes, clipped packages, repeated identities, a background-bottle regression on an old diagnostic, multiple shelf rows, and camera revisits. See `BASELINE_RESULTS.md` and `CORRECTED_MODEL_RESULTS.md` for development history.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m shelf_audit index
.venv/Scripts/python.exe -m shelf_audit serve
```

Open http://127.0.0.1:8765 and upload a JPEG or PNG containing one product. The first index build downloads the pretrained weights to `.cache/torch`; query images are processed locally and are not persisted by the demo server. CUDA is selected if available; `--device cpu` provides a fallback. Run the server on localhost only.

## Commands

Sample development video for visual review (timestamped full-resolution JPEGs, contact sheets, and HTML gallery):

```powershell
.venv/Scripts/python.exe -m shelf_audit.sample_video data/videos/pass_original_1.mov --out data/video_frames/pass_original_1 --interval 0.5
```

Choose a fresh output directory when changing the sampling interval. The sampler decodes frames sequentially, applies orientation metadata, and selects frames by their presentation timestamps. The source hash and frame indices are recorded in its manifest.

Evaluate the user-approved manual boxes on nine selected development frames against the existing reference index:

```powershell
.venv/Scripts/python.exe -m shelf_audit.evaluate_video_queries
```

This reads `shelf_audit/video_query_boxes.json` and writes crops, YOLO labels, box overlays, recognition results, and `data/video_queries/review.html`. The labels have been reviewed and approved by the user. This measures recognition with manual boxes; automatic detection and tracking are subsequent stages. Video crops are never added to the reference index.

```powershell
.venv/Scripts/python.exe -m shelf_audit predict path/to/product_crop.jpg
.venv/Scripts/python.exe -m shelf_audit evaluate
.venv/Scripts/python.exe -m unittest discover -s tests
```

`index` reads `data/catalog_crops/manifest.json`, hashes reference images and saves `artifacts/catalog_index`. `predict` returns the three closest distinct product identities with their supporting reference image. `evaluate` writes `outputs/reference_evaluation.json`, excluding the query and byte-identical copies from retrieval. Index loading rejects changed catalog photos and incompatible preprocessing.

## Automatic video baseline

Train the single-class detector and process a saved video:

```powershell
.venv/Scripts/python.exe -m shelf_audit.detector --epochs 40
.venv/Scripts/python.exe -m shelf_audit.process_video data/videos/pass_original_1.mov --out outputs/video_original_1_baseline
```

Use a new output directory for each processing run. Open its `review.html` for an H.264 video preview and sample images. `results.json` contains source-resolution boxes, timestamps, detector scores, and recognizer matches for every decoded frame. The preview omits audio and uses the video's nominal frame rate; original timestamps remain in JSON. Amber labels are uncertain recognizer guesses, not confirmed identities.

The detector follows [Torchvision's fine-tuning approach](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html): COCO-pretrained Faster R-CNN with MobileNetV3 FPN, replaced two-class head (background/product), frozen backbone, and trained proposal/box heads. This baseline trains for 40 epochs on seven approved original-video frames, with brightness variation and horizontal flips. Frames 60 and 270 are excluded from training and used for a small development check. The split remains correlated within one video, so it is not an independent evaluation. Model and training report are in `artifacts/product_detector`.

This stage performs automatic detection and recognition. Tracking, unique inventory counts, and global shelf-order reconstruction are not implemented yet. The three rearranged videos remain held out.

## Recognition details

### Offline tracking and shelf order

Optional refinement before tracking:

```powershell
.venv/Scripts/python.exe -m shelf_audit.refine_predictions path/to/results.json outputs/new_run/refined.json
.venv/Scripts/python.exe -m shelf_audit.track_video outputs/new_run/refined.json --out outputs/new_run/tracking --reconnect
```

Cleanup preserves excluded proposals for audit. Reconnection uses both identity and spatial overlap. It does not establish unique inventory counts. The latest evaluation hub is `outputs/final_evaluation/review.html`; all three rearranged videos have now been evaluated. Earlier statements that they remain held out describe the historical development stages.

```powershell
.venv/Scripts/python.exe -m shelf_audit.track_video outputs/video_original_1_expanded/results.json --out outputs/shelf_order_original_1_v2
.venv/Scripts/python.exe -m shelf_audit.render_tracks data/videos/pass_original_1.mov outputs/shelf_order_original_1_v2
```

Choose an empty output directory for a fresh run. This consumes automatic predictions, associates boxes using spatial overlap and estimated translation with gaps up to 0.25 seconds, and votes across accepted recognizer identities. It does not merge tracks simply because their product identities match. Tracks need five detections spanning at least 0.15 seconds; stable identity also requires five accepted votes and 70% agreement. These are development heuristics, not calibrated confidence.

Pairs visible together supply left/right constraints in 0.2-second bins. A topological ordering flags cycles and ambiguous placement. The report separates the candidate order of identified packages from unresolved tracks; it is not a validated unique-item count. The renderer uses identities from the entire saved clip, including later observations. Multi-row shelves, long-gap reidentification, camera revisits, and severe occlusion remain limitations.

The original-video result contains 12 stable identified tracks in a consistent order, one unresolved persistent track, and 27 brief tracks. The unknown crop appears to be a second detection of the cereal side panel, but it remains flagged. All 19 tests pass. Rearranged videos remain held out.

### Expanded detector experiment

```powershell
.venv/Scripts/python.exe -m shelf_audit.detector --extra-boxes shelf_audit/detector_extra_boxes.json --checkpoint artifacts/product_detector/expanded.pt
.venv/Scripts/python.exe -m shelf_audit.process_video data/videos/pass_original_1.mov --out outputs/video_original_1_expanded --checkpoint artifacts/product_detector/expanded.pt
```

This adds 10 manually annotated boxes from frames 30, 90, and 210, including packages near the background bottle. These extra annotations were visually checked by the assistant; they are not recorded as user approved. Frames 60 and 270 remain excluded from training. Training refuses to overwrite an existing checkpoint; choose a fresh checkpoint and output directory for another experiment. New training reports use the checkpoint filename with `.report.json`. The original baseline remains available.

Because the reserved development results informed this improvement, they are a tuning diagnostic, not an untouched test. Independent validation still requires the held-out videos.

1. Correct image orientation, convert to RGB, pad to a square with the ImageNet mean color, and resize to 224Ã—224. Padding preserves the whole package rather than cutting off edges. Use ImageNet normalization. This intentionally differs from the weights' default center-crop pipeline and is held constant for reference/query images.
2. Extract the frozen CNN's global pooled features and normalize to unit length.
3. Compute cosine similarity to every reference. Each product's score is its highest reference similarity; return the top three distinct products.
4. Flag low scores or a small gap between the top two products for review. Defaults (0.65 similarity, 0.03 margin) are provisional, not calibrated; do not interpret scores as probabilities or promise reliable unknown-product rejection.

Reference-only leave-one-image-out results share the same products, physical packages, background and recording session. They are development diagnostics, not independent retail accuracy. Do not use the same reference photo as both query and evidence when reporting performance. The three videos named `original` are designated development footage. The three rearranged videos remain held out; no evaluation videos are used by these commands.

Data, generated indexes, model weights, and local environments are gitignored. The catalog's open packages and household background may differ from store footage. Validate using manually checked product crops from development video before tuning thresholds or choosing fine-tuning work.

Model sources: [Torchvision ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), [PyTorch installation](https://pytorch.org/get-started/previous-versions/).
