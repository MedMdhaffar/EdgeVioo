# Abuse hard-negative training experiment

## Question

Does explicitly penalizing abuse scores on clips labeled fighting or shooting improve abuse ranking beyond the existing rare-class oversampling run?

## Setup

- Used the source-group-held-out split in `data/lists_abuse_sourceholdout_20261007/` (3,040 train clips, including 40 abuse positives; 764 validation clips, including 10 abuse positives; source groups do not overlap).
- Kept the official test split out of training and this experiment's evaluation.
- Started from the same 30-epoch cross-fusion recipe, seed 43, and `class_over_sample=auto` as the prior oversampling-only run.
- Changed one setting: for the abuse output, multiply negative loss by **3** on training clips labeled fighting and/or shooting. Clips also labeled abuse are excluded from that hard-negative weighting.
- The trainer selects `ckpt_best.pt` by overall binary validation AP, matching the prior runs. The per-category diagnostic was then computed on the source-held-out validation CSV.

## Results

| Source-validation measure | Baseline | Oversampling | Oversampling + hard-negative weight 3 |
|---|---:|---:|---:|
| Abuse AP | 0.0629 | **0.1349** | 0.1132 |
| Mean abuse score on positives (10 clips) | 0.150 | **0.473** | 0.212 |
| Mean abuse score on negatives (754 clips) | 0.0398 | 0.0917 | **0.0462** |
| Abuse positives above 0.5 | 0/10 | **4/10** | 0/10 |
| Abuse negatives above 0.5 | 1/754 | 29/754 | **5/754** |

Hard-negative weighting suppressed scores on negatives, including many fighting/shooting clips, but also suppressed abuse-positive scores. Its abuse ranking (AP) was lower than oversampling alone, and it detected no held-out abuse positives at the fixed 0.5 threshold. The oversampling-only checkpoint remains the better of these three for abuse ranking on this validation split.

## Interpretation and limits

This is evidence against using a uniform 3x penalty on all fighting/shooting negatives. It does not establish that hard negatives are unhelpful in general: validation has only 10 abuse positives, the tags are weak clip-level labels, and one seed was used. Do not promote this checkpoint based on its lower negative scores. The current data still has only 50 labeled abuse training clips overall; this run used 40 after the source-group holdout. No new positive examples were added, because no additional verified abuse annotations are currently in the project.

## Artifacts

- Run/checkpoint: `runs/2026-10-07_p3sourceval_cross_osample_hardneg3_s43/`
- Training history and metrics: `history.csv`, `metrics.json`
- Held-out per-class diagnostic: `sourceval_class_diagnostic.json`
- Comparison for the earlier baseline vs oversampling: `data/lists_abuse_sourceholdout_20261007/comparison.json`
