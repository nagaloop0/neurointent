import numpy as np
import pytest

from neurointent.data import synthetic
from neurointent.evaluation import evaluate, splits


def test_fixture_reproducible():
    a, b = synthetic(7, 10), synthetic(7, 10)
    np.testing.assert_array_equal(a.X, b.X)
    assert a.X.shape == (30, 8, 480)


def test_no_recording_leakage():
    data = synthetic(trials_per_run=10)
    seen = []
    for train, test in splits(data):
        assert set(data.groups[train]).isdisjoint(data.groups[test])
        seen.extend(test)
    assert sorted(seen) == list(range(len(data.y)))


def test_end_to_end_signal_recovery():
    report = evaluate(synthetic(trials_per_run=12))
    assert report["models"]["majority"]["balanced_accuracy"] == 0.5
    assert report["models"]["csp_lda"]["balanced_accuracy"] > 0.8
    assert sum(map(sum, report["models"]["csp_lda"]["confusion_matrix"])) == 36


def test_invalid_groups_rejected():
    data = synthetic(trials_per_run=10)
    data.groups[:] = 1
    with pytest.raises(ValueError, match="independent"):
        splits(data)
