# UCF-Crime Abuse augmentation experiment

## Question

Does adding UCF-Crime's officially training-split Abuse videos improve XD-Violence abuse detection on an XD source-heldout validation split?

## Data and protocol

- Added the 48 UCF-Crime Abuse videos in the official training split. The two Abuse videos assigned to UCF-Crime's official test split were excluded.
- Used available CLIP ViT-B/32 mean+standard-deviation visual features and log-mel audio features. `Abuse002_x264` has no audio stream; the existing multimodal dataset loader supplied its missing-audio fallback.
- Compared two 30-epoch runs with the same model, seed (43), oversampling, optimizer, augmentation, and training settings. The only data difference was the 48 supplementary UCF clips.
- Both runs used the same XD source-heldout validation set: 672 clips, including 10 abuse positives. UCF clips were training-only.
- The official XD-Violence test set was not used.
- The validation checkpoints were selected by overall binary AP, not abuse AP. Diagnostics below use those selected checkpoints.
- UCF-Crime's [official category definition](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) is broader than the project's intended class and the clips were not individually confirmed by continuous playback. The augmentation therefore tests weak labels, not 48 verified project-specific abuse instances.

## Results

| Model | Best epoch | Overall validation AP | Abuse AP | Heldout abuse positives at score >= 0.5 | Heldout negatives at score >= 0.5 |
|---|---:|---:|---:|---:|---:|
| XD-only baseline | 26 | 0.9883 | 0.1469 | 1 / 10 | 8 / 662 |
| XD + 48 UCF train clips | 16 | 0.9873 | 0.1095 | 4 / 10 | 38 / 662 |

On the selected augmented checkpoint, mean abuse score for the 10 positives rose from 0.252 to 0.414, but mean score on the negatives also rose from 0.056 to 0.118. The model found more heldout positives at a fixed 0.5 threshold, with a substantial increase in false positives and lower abuse ranking AP. The final-epoch abuse APs were 0.1456 (XD-only) and 0.1323 (augmented), also not an improvement.

## Conclusion

Do not adopt this UCF augmentation as the current training recipe. These 48 clips did not improve abuse ranking on the matched XD source-heldout validation set. The threshold recall increase comes with many more false alerts, consistent with UCF's broader category definition and possible label mismatch. Keep the videos as a local candidate pool; screen and relabel clips against the project definition before a later, narrower experiment.

The validation set contains only 10 abuse positives, so the measured differences are directional rather than a stable estimate of real-world performance. No claim about official-test performance is made.

## Reproduction artifacts

- Baseline run: `runs/2026-10-08_ucf_abuse_sourceval_baseline_s43/` (`config.json`, `metrics.json`, checkpoints, and `sourceval_class_diagnostic.json`).
- Augmented run: `runs/2026-10-08_ucf_abuse_sourceval_augmented_s43/` (same artifacts).
- Matched baseline lists: `data/lists_abuse_clip_sourceholdout_20261008_baseline/`.
- Augmented training lists: `data/lists_abuse_clip_sourceholdout_20261008_ucf_augmented/`.
- UCF video provenance and official split membership: `data/raw/ucf_crime/metadata/ucf_crime_training_provenance.md` and `abuse_manifest.json`.
- Clip-by-clip review tracker: `sheets/abuse_audit/ucf_crime_manual_review.csv`. Set `project_label` to `abuse`, `not_abuse`, or `uncertain` after watching the full clip; use `notes` for context. Do not treat the original UCF label as confirmed for this project.
