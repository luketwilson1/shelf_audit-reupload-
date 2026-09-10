# Corrected-box detector candidate

Trained `artifacts/product_detector/corrected.pt` for 40 epochs on 19 frames / 65 boxes: the previous 10 development frames plus nine reviewed problem frames from rearranged videos 2 and 3. The new 35 box coordinates were annotated by the assistant; the user confirmed the product identities in the failure crops, not these precise coordinates. Original two-frame development check remains excluded from training. Fresh video was not accessed.

| Diagnostic | Previous detector | Corrected candidate |
|---|---:|---:|
| Matches on 35 corrected training boxes, IoU >= 0.5 | 35 | 35 |
| Extra predictions on those frames | 6 | 0 |
| Matches on original reserved seven boxes | 7 | 7 |
| Extra predictions on original reserved frames | 0 | 1 |
| Persistent unknown tracks on full video 3 | 5 | 0 |
| Stable identified tracks on video 3 | 12 | 12 |

Detector threshold remained 0.5; recognizer, overlap cleanup, and reconnection settings stayed fixed. The original-frame regression is a background bottle detection. The corrected frames and neighboring video frames now share training data, so these results describe development fit, not independent accuracy. A fresh video is needed to assess generalization. Keep the previous model available for comparison; this candidate has not replaced it automatically.

Reproduce training after materializing frames listed in `data/corrected_detector/manifest.json`:

```powershell
.venv/Scripts/python.exe -m shelf_audit.detector --extra-boxes shelf_audit/detector_extra_boxes.json --corrections shelf_audit/corrected_boxes.json --checkpoint artifacts/product_detector/corrected.pt
```

The command refuses to overwrite existing checkpoints. Choose a new filename for another run. Frame manifest records source-video hashes, frame indices and image hashes. Annotation gallery: `data/corrected_detector/review.html`. Tracking preview: `outputs/corrected_development/video3/tracking/review.html`. Raw diagnostics: `outputs/corrected_development/comparison.json` and `artifacts/product_detector/corrected.report.json`.

All 24 tests passed. Earlier model files and evaluation predictions were preserved. Videos 2 and 3 are now explicitly marked as training/development sources; their historical frozen evaluations remain available.
