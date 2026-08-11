"""
Predict odor properties (odorous/odorless and olfactory threshold)
from SMILES using the pre-trained POVDB model.

Usage:
    # Single molecule
    python scripts/predict.py --smiles "CC(=O)C"

    # Batch prediction from a CSV file with a SMILES column
    python scripts/predict.py --csv examples/example_data.csv \
        --smiles-column SMILES --output results.csv

    # Read SMILES from stdin (one per line)
    echo "CCSCC" | python scripts/predict.py
"""

import os
import sys
import argparse
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as `python scripts/predict.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from povdb import OdorPredictor


def print_result(result):
    """Print a single prediction result."""
    if result["is_odorous"] is None:
        print(f"{result['smiles']}\tINVALID_SMILES")
        return
    label = "odorous" if result["is_odorous"] else "odorless"
    thr = result["threshold_ppm"]
    if result["is_odorous"]:
        print(f"{result['smiles']}\t{label}\t"
              f"threshold={thr:.3g} ppm (bin {result['threshold_bin']})")
    else:
        print(f"{result['smiles']}\t{label}\t"
              f"threshold>=1 ppm (bin {result['threshold_bin']})")


def main():
    parser = argparse.ArgumentParser(
        description="Predict odor properties from SMILES.")
    parser.add_argument("--smiles", help="Single SMILES string.")
    parser.add_argument("--csv", help="CSV file with SMILES column.")
    parser.add_argument("--smiles-column", default="SMILES",
                        help="Column name for SMILES (default: SMILES).")
    parser.add_argument("--model-dir", default="models",
                        help="Directory of pre-trained model "
                             "(default: models).")
    parser.add_argument("--output",
                        help="Write batch results to this CSV file.")
    args = parser.parse_args()

    predictor = OdorPredictor(model_dir=args.model_dir)

    if args.smiles:
        print_result(predictor.predict(args.smiles))

    elif args.csv:
        data = pd.read_csv(args.csv)
        if args.smiles_column not in data.columns:
            raise ValueError(f"Column '{args.smiles_column}' not found in "
                             f"{args.csv}. Available: {list(data.columns)}")
        results = [predictor.predict(s) for s in data[args.smiles_column]]
        df = pd.DataFrame(results)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Results written to {args.output}")
        else:
            print(df.to_string(index=False))

    else:
        # Read SMILES from stdin
        for line in sys.stdin:
            line = line.strip()
            if line:
                print_result(predictor.predict(line))


if __name__ == "__main__":
    main()
