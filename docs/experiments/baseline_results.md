# Reference recognition baseline

Frozen ResNet-50 IMAGENET1K_V2 encoder; 2,048-dimensional L2-normalized vectors; maximum reference cosine similarity per product. Full-package square padding and 224×224 input.

| Diagnostic | Result |
|---|---:|
| Reference crops | 56 |
| Product identities | 12 |
| Leave-one-image-out top-1 matches | 56 / 56 |
| Leave-one-image-out top-3 matches | 56 / 56 |
| Candidate matches at provisional thresholds | 53 / 56 |
| Flagged for review | 3 / 56 |

Each query and any byte-identical copies were excluded from its candidate references. The encoder was not trained on these product images. The diagnostic still shares physical packages, products, background, and capture session; it does not measure independent video accuracy. The three review flags are not classification errors: their top choice was correct, but similarity or margin did not pass the provisional thresholds.

Minimum cosine similarity 0.65 and minimum top-two class margin 0.03 were not calibrated against unknown products. These are review heuristics, not probabilities or validated open-set rejection.

The initial GPU index build embedded all 56 crops in 3.318 seconds, including image loading, preprocessing, and first inference overhead, but excluding model loading. This is a one-run development measurement, not a steady-state latency benchmark.

## Development video recognition

Nine selected frames from `pass_original_1.mov` yielded 27 manually boxed queries covering all 12 products. Boxes and identities were annotated by the assistant and approved by the user. The reference catalog and thresholds stayed unchanged.

| Diagnostic | Result |
|---|---:|
| Top-1 matches | 25 / 27 |
| Top-3 matches | 26 / 27 |
| Top-1 on packages without border clipping | 10 / 10 |
| Flagged for review | 8 / 27 |
| Correct among accepted candidates | 19 / 19 |

Both incorrect top matches were narrow border fragments (coffee and rice), and both were flagged for review. These are recognition results given manual localization, not detector or tracking accuracy. Selected frames are correlated and show the same physical packages and household setting. Unknown-product rejection remains untested. All three rearranged videos remain held out.

Local review: `data/video_queries/review.html`; detailed outputs and annotations are alongside it. Reproduce with `.venv/Scripts/python.exe -m shelf_audit.evaluate_video_queries` after sampling the development video at the default interval.

## First automatic detector

Single-class COCO-pretrained Faster R-CNN MobileNetV3 FPN; frozen backbone; proposal and box heads trained for 40 epochs on seven approved development frames (20 boxes). Two reserved frames (60 and 270, seven boxes) were not used for gradient updates. This is a correlated same-video check, not an independent benchmark. No checkpoint or threshold selection was performed using these results.

At detector score >= 0.5 and one-to-one IoU >= 0.5 matching, the reserved frames yielded 5 true positives, 1 false positive, and 2 misses: precision 83.3%, recall 71.4%. These tiny-sample detector metrics are distinct from the manually cropped recognizer results above. The false detection includes a background bottle; partial packages are missed. More development-video annotations are needed before final evaluation.

The saved detector then processed all 398 frames of `pass_original_1.mov` with the unchanged catalog recognizer. `outputs/video_original_1_baseline/review.html` provides the automatic H.264 preview, sample frames, and per-frame JSON. Processing took 42.43 seconds excluding model loading on CUDA, including decoding, inference, and video encoding. This is one run, not a real-time performance claim. The complete preview includes training frames and cannot be treated as an accuracy test.

Verification: 13 unit tests passed, the preview decoded to 398 frames, all prediction boxes were within source-image bounds, and source timestamps were monotonic. Tracking, unique counts, and global shelf order remain subsequent work. The held-out rearranged videos and reference embeddings were unchanged.

## Expanded detector experiment

## Overlap cleanup and reconnection evaluation

Added conservative uncertain-proposal cleanup: remove a contained fragment only when an accepted package explains it, or remove a merged proposal only when two separate accepted boxes explain most of it. Excluded detections remain in JSON. This does not invent split boxes or retrain the detector. Track reconnection up to 0.75 seconds requires consistent accepted identity plus spatial overlap; simultaneous same-SKU detections remain separate.

Development run: 19 per-frame proposals excluded, short tracks reduced from 27 to 25, all 12 identified products retained in order; one persistent unknown remained. All 24 tests passed. This demonstrates limited improvement, not resolution of the failure class.

The version was frozen before evaluating the two remaining rearranged videos. Video 2: 298 frames, 12 distinct accepted identities across 13 identified tracks (breadcrumbs repeated), four persistent unknown tracks. Video 3: 305 frames, 12 distinct accepted identities across 12 identified tracks, five persistent unknown tracks. Both order reports require review. No frame-level accuracy or unique inventory count is claimed. All tracked preview frames decoded and frozen hashes verified. First rearranged evaluation remains unchanged. All three rearranged clips have now been evaluated; future changes require fresh footage for another untouched test.

Review hub: `outputs/final_evaluation/review.html`. Frozen file hashes: `outputs/final_evaluation/frozen_manifest.json`.

Added 10 assistant-reviewed boxes from three original-video frames (30, 90, 210). The background bottle remains unlabeled so it provides a background example. Training now uses 10 frames with 30 boxes; the original seven validation boxes in frames 60 and 270 are unchanged and excluded from training. Training schedule and detection thresholds are unchanged.

The reserved-frame result improved from 5 true positives, 1 false positive, and 2 misses to 7 true positives, 0 false positives, and 0 misses. The bottle detection is absent at frame 60. This is not a claim of perfect detector accuracy: these are two correlated development frames, and their earlier failures informed the additional training data.

Checkpoint: `artifacts/product_detector/expanded.pt`; report: `artifacts/product_detector/expanded.report.json`. Additional annotation review: `outputs/detector_annotation_review/review.html`. Updated automatic video: `outputs/video_original_1_expanded/review.html`. Original baseline artifacts are preserved.
