"""Data sources with explicit provenance and recording groups."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metadata: dict


def synthetic(seed=42, trials_per_run=40):
    """Deliberately separable oscillatory fixture; NOT physiological evidence."""
    rng = np.random.default_rng(seed)
    sfreq = 160
    time = np.arange(480) / sfreq
    X, y, groups = [], [], []
    mixing = rng.normal(size=(8, 8))
    mixing, _ = np.linalg.qr(mixing)
    for run in (4, 8, 12):
        labels = np.arange(trials_per_run) % 2
        rng.shuffle(labels)
        for label in labels:
            sources = rng.normal(size=(8, len(time)))
            phase = rng.uniform(0, 2 * np.pi, 2)
            sources[0] += (1.0 if label else 2.5) * np.sin(2 * np.pi * 10 * time + phase[0])
            sources[1] += (2.5 if label else 1.0) * np.sin(2 * np.pi * 12 * time + phase[1])
            X.append(mixing @ sources * 1e-6)
            y.append(label)
            groups.append(run)
    sos = butter(4, [8, 30], fs=sfreq, btype="bandpass", output="sos")
    return Dataset(sosfiltfilt(sos, np.array(X), axis=-1), np.array(y),
                   np.array(groups), {"source": "synthetic", "seed": seed,
                   "sfreq": sfreq, "warning": "Synthetic sanity check; not real EEG performance."})


def physionet(subject=1, cache="data"):
    """Load only left/right imagined fist runs; keep run boundaries intact."""
    import mne
    from mne.datasets import eegbci

    if not 1 <= subject <= 109:
        raise ValueError("subject must be between 1 and 109")
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    arrays, labels, groups = [], [], []
    counts = {}
    for run in (4, 8, 12):
        path = eegbci.load_data(subject, [run], path=str(cache), update_path=False)[0]
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        eegbci.standardize(raw)
        raw.pick("eeg")
        raw.set_eeg_reference("average", projection=False, verbose=False)
        # Filter each run independently: no signal crosses a held-out boundary.
        raw.filter(8, 30, verbose=False)
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose=False)
        epochs = mne.Epochs(raw, events, event_id={"left": 1, "right": 2},
                            tmin=1, tmax=3.5, baseline=None, preload=True,
                            reject_by_annotation=True, verbose=False)
        arrays.append(epochs.get_data(copy=True))
        labels.append(epochs.events[:, 2] - 1)
        groups.extend([run] * len(epochs))
        counts[str(run)] = {"kept": len(epochs), "events": len(events)}
    return Dataset(np.concatenate(arrays), np.concatenate(labels), np.array(groups),
                   {"source": "physionet-eegmmidb-1.0.0", "subject": subject,
                    "runs": [4, 8, 12], "sfreq": float(raw.info["sfreq"]),
                    "epoch_seconds": [1, 3.5], "bandpass_hz": [8, 30], "counts": counts,
                    "doi": "10.13026/C28G6P"})
