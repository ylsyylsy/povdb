"""
Train the olfactory threshold regression model for POVDB.

This script reproduces the pre-trained Random Forest model shipped in
``models/best_model.joblib``:

    RandomForestRegressor(n_estimators=100, random_state=42,
                          max_depth=10, min_samples_leaf=1,
                          bootstrap=True)

Trained on Morgan fingerprints (radius=2, 2048 bits) of the training
compounds.  The target label is the olfactory threshold bin (0-5), and
the physical threshold is recovered as ``threshold = 10 ** (label - 5)``
ppm.  Label 5 corresponds to odorless compounds (threshold >= 1 ppm).

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --data data/5_data_quan.csv \
        --output models/best_model.joblib
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
import joblib

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string to a fixed-length Morgan fingerprint."""
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return [int(b) for b in fp.ToBitString()]
    except Exception:
        return None


def get_training_data(data_path):
    """Load and prepare training data (SMILES + Label columns)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} training records from {data_path}")

    data["fingerprint"] = data["SMILES"].apply(smiles_to_morgan)
    data = data.dropna(subset=["fingerprint"])

    X = np.array(data["fingerprint"].tolist())
    y = data["Label"].values

    print(f"Prepared training data: X {X.shape}, y {y.shape}")
    print(f"Label distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train the POVDB olfactory threshold model.")
    parser.add_argument("--data", default="data/5_data_quan.csv",
                        help="Path to training CSV (SMILES,Label columns).")
    parser.add_argument("--output", default="models/best_model.joblib",
                        help="Output path for the trained model.")
    args = parser.parse_args()

    X, y = get_training_data(args.data)

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10,
        min_samples_leaf=1, bootstrap=True)
    model.fit(X, y)

    # Cross-validated performance on the training data (in log10 units)
    y_cv = cross_val_predict(model, X, y, cv=5)
    mae = mean_absolute_error(y, y_cv)
    print(f"Cross-validated MAE (label units): {mae:.4f}")
    print(f"  -> equivalent to {10 ** mae:.2f}x threshold error (geometric)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Model saved to {args.output} "
          f"({os.path.getsize(args.output)} bytes)")


if __name__ == "__main__":
    main()
