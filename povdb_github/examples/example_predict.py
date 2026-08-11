"""
Example: predict odor properties with the pre-trained POVDB model.

Run from the repository root:

    python examples/example_predict.py
"""

import os
import sys

# Make the povdb package importable when running from the examples/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from povdb import OdorPredictor
from povdb.query import POVDBQuery
from povdb.annotator import PeakAnnotator


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(repo_root, "models")

    # ---------------------------------------------------------------
    # 1. Odor property prediction
    # ---------------------------------------------------------------
    print("=" * 60)
    print("1. Odor property prediction (SMILES -> odorous + threshold)")
    print("=" * 60)

    predictor = OdorPredictor(model_dir=model_dir)

    test_smiles = [
        "CC(=O)C",        # acetone
        "CCSCC",          # diethyl sulfide
        "CCCCCCCC",       # octane
        "CCO",            # ethanol
    ]

    for smiles in test_smiles:
        result = predictor.predict(smiles)
        status = ("odorous" if result["is_odorous"] else "odorless")
        thr = result["threshold_ppm"]
        if result["is_odorous"]:
            print(f"  {smiles:<14} -> {status:<9} threshold={thr:.3g} ppm")
        else:
            print(f"  {smiles:<14} -> {status:<9} threshold>=1 ppm")

    # Batch prediction from CSV
    csv_path = os.path.join(repo_root, "examples", "example_data.csv")
    if os.path.exists(csv_path):
        print("\nBatch prediction from examples/example_data.csv:")
        df = predictor.predict_from_csv(csv_path)
        print(df.to_string(index=False))

    # ---------------------------------------------------------------
    # 2. POVDB spectral library search
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. POVDB spectral library search")
    print("=" * 60)

    db = POVDBQuery()
    print(f"\nRecords in sample library: {len(db.records)}")

    hits = db.query_by_name("tridecanone")
    print(f"\nQuery by name 'tridecanone': {len(hits)} hit(s)")
    for rec in hits[:3]:
        print(f"  {rec.get('NAME')} | CID={rec.get('CID')} | "
              f"FORMULA={rec.get('FORMULA')}")

    sims = db.find_similar("CC(C)(C)C", threshold=0.6)
    print(f"\nSimilar to neopentane (MACCS Tanimoto >= 0.6): "
          f"{len(sims)} hit(s)")
    for rec, sim in sims[:3]:
        print(f"  {rec.get('NAME')}  Tanimoto={sim:.3f}")

    # ---------------------------------------------------------------
    # 3. Peak annotation (odor contribution)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Peak annotation with odor contribution")
    print("=" * 60)

    # Build a small in-memory peak table for demonstration
    import pandas as pd
    demo_table = pd.DataFrame({
        "Name": ["Acetone", "Diethyl sulfide", "Unknown"],
        "SMILES": ["CC(=O)C", "CCSCC", None],
        "Peak Area": [1.0e6, 5.0e4, 8.0e5],
    })
    demo_table.to_csv("_demo_peaks.csv", index=False)

    annotator = PeakAnnotator(povdb_path=None, model_dir=model_dir)
    annotated = annotator.annotate(
        "_demo_peaks.csv",
        smiles_column="SMILES",
        area_column="Peak Area")
    print(annotated.to_string(index=False))

    summary = annotator.summarize(annotated)
    print(f"\nSummary: {summary}")
    os.remove("_demo_peaks.csv")


if __name__ == "__main__":
    main()
