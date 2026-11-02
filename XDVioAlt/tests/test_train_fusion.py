"""Focused tests for source-safe sampling controls in fusion training."""
from safewatch.train_fusion import hard_fighting_mask


def test_hard_fighting_mask_requires_matching_source_and_fighting_label():
    rows = [
        {"clip_id": "Ip.Man.2008__#fight", "binary": "1", "labels": "fighting"},
        {"clip_id": "Ip.Man.2008__#normal", "binary": "0", "labels": ""},
        {"clip_id": "Taken.3.2014__#fight", "binary": "1", "labels": "fighting"},
        {"clip_id": "Other__#shoot", "binary": "1", "labels": "shooting"},
    ]
    selected = hard_fighting_mask(rows, "Ip.Man.2008, Taken.2.UNRATED.EXTENDED.2012")
    assert selected.tolist() == [True, False, False, False]
