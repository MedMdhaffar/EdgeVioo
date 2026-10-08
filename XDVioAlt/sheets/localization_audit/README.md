# Manual localization validation set

This review sample was drawn only from the source-heldout validation partition of XD-Violence training clips (`data/lists_abuse_sourceholdout_20261007/val.csv`). It does not include the official test clips or their temporal annotations.

## How to annotate

1. Watch each full video at the `video_path` in `validation_annotation_template.csv`.
2. Mark every incident interval in seconds in `incident_intervals_seconds`, using `start-end;start-end` for multiple events (example: `12.4-18.7;41.0-43.2`). Leave blank if no incident is visible/audible and set `review_status` to `reviewed_no_event`.
3. Set `review_status` to `reviewed_event` when intervals are entered, or `uncertain` if you cannot determine the boundaries.
4. Put corrected labels in `reviewed_labels` and context in `notes`. The XD labels are weak clip-level labels, not confirmed ground truth.

Use the full clip context. Do not use model-highlighted predictions to choose boundaries; annotate independently to reduce confirmation bias. After review, return the filled CSV. Segment threshold and gap tuning can then use this validation set; final official-test metrics remain untouched.

## Sampling

Deterministic seed: 20261008. Target quotas: 10 clips per each of six anomaly categories plus 20 normal controls. Clips are deduplicated across overlapping category strata. Selected clips: 80 (20 controls). Raw files were checked locally; all selected clips have readable duration metadata.
