import argparse
import json
from pathlib import Path

from .data import physionet, synthetic
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="NeuroIntent offline EEG benchmark")
    parser.add_argument("--source", choices=["synthetic", "physionet"], default="synthetic")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default="data")
    parser.add_argument("--output", default="artifacts/results.json")
    args = parser.parse_args()
    data = synthetic(args.seed) if args.source == "synthetic" else physionet(args.subject, args.cache)
    report = evaluate(data)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Source: {report['dataset']['source']} | Trials: {report['n_trials']}")
    for name, result in report["models"].items():
        print(f"{name}: balanced accuracy={result['balanced_accuracy']:.3f}, kappa={result['cohen_kappa']:.3f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
