# NeuroIntent 🧠

**An offline brain–computer interface research project that decodes imagined left- and right-hand movement from EEG.**

Built for reproducible experimentation: explicit data provenance, recording-aware evaluation, a classical spatial-filtering model, and honest baseline comparisons.

## What it does

- Loads PhysioNet EEG Motor Movement/Imagery recordings (imagined fists, runs **4, 8, 12**).
- Applies average reference and 8–30 Hz filtering independently to each recording.
- Extracts 1–3.5 second post-cue epochs, excluding rest cues.
- Compares majority-class, channel log-variance + logistic regression, and **Common Spatial Patterns + shrinkage LDA**.
- Holds out one complete run at a time. CSP, scaling, and classifiers learn only from the training runs.
- Exports balanced accuracy, Cohen’s kappa, confusion matrices, fold scores, trial counts, and environment versions to JSON.

## Quick start

Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
neurointent --source synthetic
```

Synthetic mode is a deterministic software sanity check with deliberately separable oscillations. Its score is **not evidence of performance on human EEG**.

Run the real-data experiment (requires internet for the initial download, cached locally):

```bash
neurointent --source physionet --subject 1 --output artifacts/subject-001.json
```

Repeat across subjects before claiming broad performance:

```bash
neurointent --source physionet --subject 2 --output artifacts/subject-002.json
neurointent --source physionet --subject 3 --output artifacts/subject-003.json
```

## Evaluation contract

### Verified first experiment

Subject S001: 45 trials, 64 EEG channels, three held-out-run folds. CSP + LDA achieved **64.3% balanced accuracy** (kappa 0.287), compared with **60.4%** for log-variance + logistic regression and **50.0%** for the majority baseline. This is a small, single-subject check, not evidence of population-level performance or statistical significance. Full fold scores and environment versions are in [the recorded benchmark](docs/benchmark-subject-001.json).

Labels are `0=left`, `1=right`. Confusion-matrix rows are true labels; columns are predictions. Each test trial appears in exactly one fold. Aggregate metrics use pooled out-of-fold predictions, with per-run scores reported separately. Model settings are fixed in advance; selecting a model after looking at these scores requires new independent test data.

This is **within-subject, held-out-run evaluation**, not cross-subject or cross-day validation. Filtering uses the full recording offline, including future samples. This is not a causal streaming system. There is no hardware integration, clinical validation, artifact rejection by amplitude, or claim that it can read arbitrary thoughts. EEG artifacts and subject variability remain important limitations.

## Project layout

```text
src/neurointent/data.py        real EEG loading and synthetic fixture
src/neurointent/evaluation.py  models and independent-run evaluation
src/neurointent/cli.py         reproducible command-line experiment
tests/                        reproducibility, split integrity, signal recovery
docs/                         methodology and résumé wording
.github/workflows/            automated offline checks
```

Raw EEG, local environments, and generated experiments are excluded from Git. Curated benchmark evidence can be stored under `docs/` after review. Data remains subject to its own terms; do not treat code licensing as data licensing.

## References

- [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
- [MNE: CSP motor-imagery decoding example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html). NeuroIntent uses left/right imagery runs and a held-out-run protocol; the example's task selection and validation protocol differ.

## Next milestones

1. Publish a fixed multi-subject benchmark with all subject results, including failures.
2. Compare filter-bank CSP with nested model selection.
3. Add causal preprocessing and recorded-signal replay before hardware integration.
4. Validate a real device adapter with explicit calibration and latency measurements.
