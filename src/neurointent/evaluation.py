"""All learned transforms are fitted inside each training fold."""
import platform
from importlib.metadata import version

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression


def log_variance(X):
    return np.log(np.maximum(np.var(X, axis=-1), np.finfo(float).tiny))


def models():
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logvar_logreg": make_pipeline(FunctionTransformer(log_variance), StandardScaler(),
                                        LogisticRegression(random_state=42)),
        "csp_lda": make_pipeline(CSP(n_components=4, reg="ledoit_wolf", log=True,
                                     norm_trace=False),
                                  LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    }


def splits(dataset):
    X, y, groups = dataset.X, dataset.y, dataset.groups
    if X.ndim != 3 or len(X) != len(y) or len(y) != len(groups):
        raise ValueError("Expected aligned trials × channels × samples, labels and groups")
    if not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected finite EEG and binary labels 0/1")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two independent recording groups are required")
    result = list(LeaveOneGroupOut().split(X, y, groups))
    for train, test in result:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Each training and test fold must contain both classes")
    return result


def evaluate(dataset):
    import mne
    mne.set_log_level("WARNING")
    folds = splits(dataset)
    results = {}
    for name in models():
        predictions = np.empty_like(dataset.y)
        scores = []
        for train, test in folds:
            model = models()[name]
            model.fit(dataset.X[train], dataset.y[train])
            pred = model.predict(dataset.X[test])
            predictions[test] = pred
            scores.append({"held_out_run": int(dataset.groups[test][0]),
                           "n_train": len(train), "n_test": len(test),
                           "balanced_accuracy": float(balanced_accuracy_score(dataset.y[test], pred))})
        results[name] = {
            "accuracy": float(accuracy_score(dataset.y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(dataset.y, predictions)),
            "cohen_kappa": float(cohen_kappa_score(dataset.y, predictions)),
            "confusion_matrix": confusion_matrix(dataset.y, predictions, labels=[0, 1]).tolist(),
            "folds": scores,
        }
    return {"dataset": dataset.metadata, "n_trials": len(dataset.y),
            "n_channels": dataset.X.shape[1], "class_order": ["left", "right"],
            "protocol": "leave-one-run-out; within-subject; offline; no hyperparameter search",
            "environment": {"python": platform.python_version(),
                            **{p: version(p) for p in ["numpy", "scipy", "mne", "scikit-learn"]}},
            "models": results}
