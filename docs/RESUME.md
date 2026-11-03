# Résumé and interview notes

## Project title

NeuroIntent — EEG Motor-Imagery Decoding | Python, MNE, scikit-learn

## Accurate bullet for the implemented project

- Built an offline EEG motor-imagery decoding pipeline using PhysioNet recordings, Common Spatial Patterns, and shrinkage LDA; compared three models with held-out-run evaluation and reproducible metrics.
- Added automated checks for recording split integrity, deterministic data generation, and end-to-end signal recovery, with a GitHub Actions test workflow.

Do not claim a GitHub Actions run passed until it has run on GitHub. Add numerical real-EEG results only after checking the saved experiment and naming its subject count and validation protocol. Never put the synthetic score on a résumé as EEG accuracy.

## Be ready to explain

- Why motor imagery changes spatial patterns of sensorimotor EEG rhythms.
- How CSP learns projections that distinguish class covariance structures.
- Why fitting CSP before splitting leaks label information.
- Why random windows from one recording give weaker evidence than held-out runs.
- Why a within-subject result does not establish generalization to a new person.
- Why offline zero-phase filtering cannot be used unchanged for real-time control.
