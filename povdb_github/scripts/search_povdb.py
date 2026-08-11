"""
Search the POVDB spectral library.

Query the Potential Odorous Virtual Database (MSP format) by compound
name, SMILES, formula, exact mass, or by experimental mass spectrum
matching.  Designed for the non-targeted GC-QTOF analysis workflow.

Usage:
    python scripts/search_povdb.py --name acetone
    python scripts/search_povdb.py --smiles "CC(=O)C"
    python scripts/search_povdb.py --formula C3H6O
    python scripts/search_povdb.py --mass 58.0419 --tolerance 0.005
    python scripts/search_povdb.py --spectrum my_peaks.txt --top-k 5
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from povdb.query import POVDBQuery


def print_records(records, limit=10):
    """Print record summaries."""
    if not records:
        print("No matches found.")
        return
    for i, rec in enumerate(records[:limit]):
        tan = rec.get("_tanimoto", "")
        tan_s = f"  Tanimoto={tan}" if tan != "" else ""
        print(f"[{i + 1}] {rec.get('NAME', '?')}  "
              f"CID={rec.get('CID', '?')}  "
              f"FORMULA={rec.get('FORMULA', '?')}  "
              f"SMILES={rec.get('SMILES', '?')}{tan_s}")


def main():
    parser = argparse.ArgumentParser(
        description="Search the POVDB spectral library.")
    parser.add_argument("--db", default=None,
                        help="Path to POVDB MSP file "
                             "(default: data/sample_povdb.msp).")
    parser.add_argument("--name", help="Search by compound name.")
    parser.add_argument("--smiles", help="Search by SMILES.")
    parser.add_argument("--formula", help="Search by molecular formula.")
    parser.add_argument("--mass", type=float, help="Search by exact mass.")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Mass tolerance in Da (default: 0.01).")
    parser.add_argument("--spectrum", help="Match an experimental spectrum "
                        "(file of 'mz intensity' lines).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Max hits for spectrum matching.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max records to print.")
    args = parser.parse_args()

    db = POVDBQuery(args.db)

    if args.name:
        print(f"Searching name: {args.name}")
        print_records(db.query_by_name(args.name), args.limit)
    elif args.smiles:
        print(f"Searching SMILES: {args.smiles}")
        print_records(db.query_by_smiles(args.smiles, exact=True), args.limit)
        print("\nSimilar compounds (MACCS Tanimoto >= 0.6):")
        for rec, sim in db.find_similar(args.smiles, threshold=0.6)[:args.limit]:
            print(f"  {rec.get('NAME', '?')}  Tanimoto={sim:.3f}")
    elif args.formula:
        print(f"Searching formula: {args.formula}")
        print_records(db.query_by_formula(args.formula), args.limit)
    elif args.mass is not None:
        print(f"Searching mass: {args.mass} +/- {args.tolerance} Da")
        print_records(db.query_by_mass(args.mass, args.tolerance), args.limit)
    elif args.spectrum:
        with open(args.spectrum, "r", encoding="utf-8") as f:
            peaks = [line.strip() for line in f if line.strip()]
        hits = db.search_spectrum(peaks, top_k=args.top_k)
        print(f"Spectrum matching: {len(hits)} hits")
        for i, (rec, matched, score) in enumerate(hits):
            print(f"[{i + 1}] {rec.get('NAME', '?')}  "
                  f"matched_peaks={matched}  score={score:.1f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
