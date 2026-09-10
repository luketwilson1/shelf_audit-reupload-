# Evaluation

## Data progression

1. A 56-image catalog supplied references for 12 household retail products.
2. Original video frames supplied initial detection annotations. Two original frames were kept out of gradient updates, but informed development and are not untouched test data.
3. Three rearranged videos were evaluated with recorded model/code hashes. Failures were reviewed, and selected frames from videos 2 and 3 later became training data. Their original evaluations were preserved.
4. Three fresh clips (IMG_1849–1851) were evaluated with the frozen corrected model. They were not used in training. All show one new arrangement in the same household with the same physical packages.

The fresh results are included in [`examples/evaluation_summary.json`](../examples/evaluation_summary.json). Each produced 12 identified tracks and 12 distinct identities, with zero persistent unknown tracks. Short tracks remained: 8, 13, and 8 respectively. Two order reports retained conservative evidence flags. The owner said the results looked good; this is general visual acceptance, not exhaustive instance annotation.

## What is measured

- Earlier manual-crop recognition: 25/27 top matches, conditional on manually drawn boxes.
- Corrected-frame detector fit: 35/35 matches, extra detections falling from six to zero after those frames were added to training.
- Old seven-box diagnostic: seven matches retained, but a background-bottle false positive returned.
- Fresh-video output: track identities, persistent unknowns, order evidence, and decoded-frame integrity.

Do not combine these into a single accuracy percentage. The crop diagnostic, training fit, and fresh-video outputs measure different things. No mAP, exhaustive per-frame precision/recall, identity-switch benchmark, or unique physical-item counting accuracy was measured on fresh footage.

## Failure analysis

Unresolved tracks included thin cereal or stuffing side panels, flour clipped at the image edge, and boxes spanning multiple packages plus background. Human review established that these were views of existing products rather than additional inventory. Correcting such errors can improve training fit while introducing regressions elsewhere, as the bottle example demonstrates.

## Next independent test

Use new footage after freezing the next model. Label full-frame instances, count physical packages, verify shelf order, and include repeated SKUs, unknown products, changing backgrounds, and occlusion. Reusing reviewed failures for development is valid; calling those same examples an independent test afterward is not.

Historical working notes are under `docs/experiments/`. They describe stages as they existed at the time and may reference local, untracked artifacts.
