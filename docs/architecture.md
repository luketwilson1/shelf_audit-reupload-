# Architecture and decisions

## Separate localization from identity

The detector learns one foreground class: package. Recognition is a separate catalog lookup, allowing reference images to change without replacing the detector's classification head. The current catalog has only 12 identities; performance across thousands of retail SKUs has not been demonstrated.

The corrected detector uses 19 training frames and 65 boxes. Its backbone is frozen while proposal and box heads are trained for 40 epochs. Some box coordinates were annotated by the assistant after the owner confirmed the contents of failure crops; identity confirmation is not the same as approval of every coordinate.

## Preserve the whole crop

Recognition corrects orientation, pads to a square, resizes to 224×224, and normalizes using ImageNet statistics. ResNet-50 generates a normalized 2,048-dimensional vector. Each class score is its strongest reference match. Similarity below 0.65 or a top-two margin below 0.03 triggers review. These thresholds are provisional and do not validate unknown-product rejection.

## Track physical observations

Greedy one-to-one association uses box overlap and estimated translation. Ordinary gaps are limited to 0.25 seconds. Optional reconnection extends to 0.75 seconds only with consistent accepted identity and stronger spatial overlap. Two simultaneous packages are not merged solely because their identities match.

Stable identity needs at least five accepted votes and 70% agreement, plus temporal persistence. The tracked preview uses observations from the whole clip, including future frames; it is offline smoothing, not causal real-time recognition.

## Order from relationships

Pairs of vertically overlapping products supply left/right evidence when visible together. Evidence is aggregated into 0.2-second bins to reduce repeated-frame influence. A directed graph supplies a topological order; cycles, disconnected alternatives, unknown identities, and repeated identities produce review flags. The single-row assumption is explicit.

## Retain failures

Overlap cleanup removes an uncertain fragment only when another accepted box explains it, or an uncertain merged proposal when two separate accepted boxes explain it. Removed predictions stay in JSON. This is conservative filtering, not a learned box-splitting model.

Human corrections are stored separately from predictions. A visually satisfactory order is not a verified inventory count. Brief tracks, missed objects, and false positives can still exist beneath a clean summary.

## Local application

The application accepts one video job at a time, runs subprocess stages, polls job status, and serves playback using HTTP byte ranges. Uploads persist locally. The server binds to loopback and rejects cross-origin upload requests. It is a local development app, not a hardened public service.
