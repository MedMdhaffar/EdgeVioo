# XD-Violence abuse-label audit (initial visual screen)

## Scope and method

- Checked the official training list and confirmed all **50** abuse-labeled rows have matching extracted MP4 clips.
- Generated contact sheets with four frames sampled uniformly from each of the 50 training clips, then denser sheets (up to 24 frames) for the eight lowest-scoring held-out abuse positives and eight highest-scoring abuse false positives from the untouched official test split.
- This is a **frame-based visual screen**, not continuous playback, audio review, or expert reannotation. Borderline cases need full-context human review.

Contact sheets: [clips 1–5](train_contact_sheets/train_abuse_01_05.jpg), [6–10](train_contact_sheets/train_abuse_06_10.jpg), [11–15](train_contact_sheets/train_abuse_11_15.jpg), [16–20](train_contact_sheets/train_abuse_16_20.jpg), [21–25](train_contact_sheets/train_abuse_21_25.jpg), [26–30](train_contact_sheets/train_abuse_26_30.jpg), [31–35](train_contact_sheets/train_abuse_31_35.jpg), [36–40](train_contact_sheets/train_abuse_36_40.jpg), [41–45](train_contact_sheets/train_abuse_41_45.jpg), [46–50](train_contact_sheets/train_abuse_46_50.jpg).

Highest false positives: [test false-positive contact sheet](test_top_abuse_false_positives.jpg). The raw model diagnostics are [baseline](official_test_baseline_diagnostic.json) and [oversampling](official_test_oversampling_diagnostic.json).

Detailed dense-review notes: [selected positives and false positives](dense_review/review_notes.md). The reviewed frame sheets are in [dense_review](dense_review/).

Follow-up training experiment: [oversampling vs fighting/shooting hard-negative weighting](hard_negative_experiment.md).

Supplementary data experiment: [UCF-Crime Abuse augmentation vs XD-only baseline](ucf_crime_augmented_experiment.md). The 48 UCF official training clips did not improve heldout abuse AP and increased false positives; they are not adopted in the current training recipe. A [clip-by-clip review tracker](ucf_crime_manual_review.csv) is ready for confirming or rejecting their labels against the project's definition.

## Findings

- The abuse label is present in the downloaded/extracted data; no missing media was found for the 50 training examples.
- Abuse often co-occurs with another category: 12 clips also say fighting, 12 shooting, 1 riot, 1 car accident, and 1 explosion. These counts overlap.
- In the sampled frames, many examples visibly show assault, restraint, coercion, or other violent behavior. Some clips are context-dependent from still frames, so this screen cannot certify every label.
- The oversampling model's eight highest-scoring negative test clips include **six fighting clips, one normal clip, and one shooting clip**. The sampled fighting examples depict physical combat/action; those are plausible model confusions under XD-Violence's category boundary.
- The project owner confirmed the two labels are correct for abuse detection: *City of God* is abuse; *Black Hawk Down* contains a gunfire exchange but no abuse, so it is a valid abuse-negative. This reinforces the distinction between abuse and violence generally. YAMNet did not identify the gunfire clearly. See the [full-context review and frame sheets](dense_review/review_notes.md).
- On the official test, the baseline has abuse AP **0.1324** and oversampling **0.1319** (from the saved run metrics). At a fixed 0.5 threshold, the diagnostic shows baseline 4/11 abuse positives and 57 negatives above threshold; oversampling 2/11 and 36. The threshold tradeoff changes, but abuse ranking on the official test did not improve.

## Assessment

The immediate issue does **not** look like a download or file-extraction failure. The main visible issue is separation between the dataset's abuse label and adjacent violent categories, especially fighting. This is a preliminary diagnosis: a human reviewer should watch the full clips, particularly abuse-only clips and high-scoring fighting/normal false positives, before changing labels or expanding training data.
