# Audio feature integrity audit — 2026-10-08

Automated with [`scripts/audit_audio_features.py`](../scripts/audit_audio_features.py). The audit
checks feature-file coverage, loadability, matrix shape, expected feature width, finite values,
grid alignment, and the saved audio RMS signal-quality measurements.

| Feature set | Lists and grid | Clips | Structural issues | Signal-quality flags | Unlisted feature files |
|---|---|---:|---:|---:|---:|
| Official VGGish | `data/lists_official/{train,val,test}.csv`, snippet grid, 16-frame stride | 4,754 | 0 | 4 silent clips; 4 missing quality-stat files | 0 |
| Log-mel | `data/lists/{train,val,test}.csv`, 0.96 s patch grid, 64-frame stride | 4,750 | 0 | 4 silent clips | 47 |

Every VGGish matrix is 2D, 128-wide, finite, and on the expected snippet grid. Every log-mel matrix
is 2D, 132-wide, finite, and within the allowed patch-count tolerance for the configured 64-frame
snippet grid. The 47 log-mel files outside its lists are `Abuse###_x264` features from the
UCF-Crime augmentation.

The signal check flags a clip when every saved patch has RMS at or below **−75 dB**. The extractor's
digital-silence floor is −90 dB. The same four clips are silent in both feature sets:

- `v=7G4IvR1vQ7M__#00-00-00_00-00-45_label_B1-0-0`
- `v=87ss-JRZIs4__#1_label_G-0-0`
- `v=pZttb6pJM8U__#1_label_B1-0-0`
- `v=9CWJd1SezkA__#1_label_G-0-0`

Official VGGish also has four clips without saved quality-stat files:
`Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0`, `v=8cTqh9tMz_I__#1_label_A`,
`v=9eME1y6V-T4__#01-12-00_01-18-00_label_A`, and
`NewAdd.NBA-2017.12.25_CLE@GSW__#01-08-34_01-40-09_label_A`. Their audio feature files pass the
structural checks. Preflight reports these cases; it does not modify fusion inputs.

Detailed machine-readable results:

- [VGGish JSON audit](audio_audit_vggish_20261008.json)
- [Log-mel JSON audit](audio_audit_mel_20261008.json)

The audit detects missing files, structural grid mismatches, entirely silent signals, and constant
feature streams. Matching row counts cannot prove semantic synchronization with video; that still
requires checking source media or event-level annotations.
