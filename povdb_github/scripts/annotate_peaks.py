"""
Annotate non-targeted analysis peak tables with POVDB odor properties.

Takes an MS-DIAL / GC-QTOF peak table (with SMILES or compound names
and peak areas) and adds:

    - Is_Odorous:      odorous vs. odorless
    - Threshold_ppm:   predicted olfactory threshold
    - Odor_Contribution: (peak area) / (threshold)  -- odour potential

Usage:
    python scripts/annotate_peaks.py --input peaks.tsv \
        --smiles-column SMILES --area-column "Peak Area" \
        --output annotated.tsv
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from povdb.annotator import PeakAnnotator


def main():
    parser = argparse.ArgumentParser(
        description="Annotate peak tables with POVDB odor properties.")
    parser.add_argument("--input", required=True,
                        help="Peak table (CSV/TSV) from non-targeted "
                             "analysis (e.g. MS-DIAL export).")
    parser.add_argument("--output", default="annotated_peaks.csv",
                        help="Output file (default: annotated_peaks.csv).")
    parser.add_argument("--smiles-column", default="SMILES",
                        help="Column containing SMILES (default: SMILES).")
    parser.add_argument("--area-column", default="Peak Area",
                        help="Column containing peak areas "
                             "(default: 'Peak Area').")
    parser.add_argument("--model-dir", default="models",
                        help="Model directory (default: models).")
    parser.add_argument("--povdb", default=None,
                        help="POVDB MSP file for name-based lookup "
                             "(default: data/sample_povdb.msp).")
    args = parser.parse_args()

    annotator = PeakAnnotator(povdb_path=args.povdb, model_dir=args.model_dir)

    print(f"Annotating {args.input} ...")
    result = annotator.annotate(
        args.input,
        smiles_column=args.smiles_column,
        area_column=args.area_column)

    result.to_csv(args.output, index=False, sep="\t")
    print(f"Annotated table written to {args.output} "
          f"({len(result)} rows)")

    summary = annotator.summarize(result)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
