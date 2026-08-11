"""
Build / clean a POVDB MSP spectral library.

Maintenance pipeline for the Potential Odorous Virtual Database:

    1. Parse the MSP file into records.
    2. Remove records whose SMILES contain isotope markers
       (e.g. '[13C]', '[2H]', '[D]').
    3. (Optional) Standardize SMILES to the PubChem style.
    4. Write the cleaned library to a new MSP file.

Usage:
    python scripts/build_povdb.py --input path/to/povdb.msp \
        --output path/to/povdb_clean.msp [--standardize]
"""

import os
import re
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from povdb.utils import (parse_msp_to_dicts, records_to_msp_string,
                         remove_isotope_records, standardize_smiles)


def clean_povdb(input_file, output_file, standardize=False):
    """Parse, clean and rewrite a POVDB MSP library."""
    print(f"Step 1: parsing {input_file} ...")
    records = parse_msp_to_dicts(input_file)
    if not records:
        print("No records parsed; aborting.")
        return
    total = len(records)
    print(f"Parsed {total} records.")

    print("Step 2: removing isotope-labelled records ...")
    final_records = remove_isotope_records(records)
    removed = total - len(final_records)
    print(f"  -> removed {removed} isotope records; "
          f"{len(final_records)} remaining.")

    if standardize:
        print("Step 3: standardizing SMILES (PubChem style) ...")
        n_std = 0
        for rec in final_records:
            smi = rec.get("SMILES", "")
            std = standardize_smiles(smi)
            if std:
                rec["SMILES"] = std
                n_std += 1
            else:
                rec["SMILES"] = "NA"
        print(f"  -> standardized {n_std} records.")

    print(f"Step 4: writing {output_file} ...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    msp_string = records_to_msp_string(final_records)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(msp_string)
    print(f"Done. Clean library written to {output_file} "
          f"({os.path.getsize(output_file)} bytes).")


def main():
    parser = argparse.ArgumentParser(
        description="Build / clean a POVDB MSP library.")
    parser.add_argument("--input", required=True,
                        help="Input MSP file.")
    parser.add_argument("--output", required=True,
                        help="Output MSP file.")
    parser.add_argument("--standardize", action="store_true",
                        help="Also standardize SMILES strings.")
    args = parser.parse_args()

    clean_povdb(args.input, args.output, args.standardize)


if __name__ == "__main__":
    main()
