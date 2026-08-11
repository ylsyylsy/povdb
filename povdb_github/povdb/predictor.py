"""
Odor property predictor using pre-trained machine learning models.

Implements the two-step prediction framework described in the manuscript:

    1. Binary classification: odorous (threshold < 1 ppm) vs. odorless
       (threshold >= 1 ppm)
    2. Binning regression: quantitative olfactory threshold estimation

The pre-trained model shipped with this repository is a Random Forest
regressor trained on Morgan fingerprints that directly predicts the
threshold bin (label 0-5).  The physical olfactory threshold is recovered
as ``threshold = 10 ** (label - 5)`` ppm; label 5 corresponds to
odorless compounds (threshold >= 1 ppm).
"""

import os
import joblib
import numpy as np
import pandas as pd
from .utils import smiles_to_morgan


# Olfactory threshold bins (ppm); label = round(log10(threshold)) + 5
THRESHOLD_BINS = [
    (0, 1e-4),        # Bin 0: extremely low (< 0.0001 ppm)
    (1e-4, 1e-3),     # Bin 1: very low
    (1e-3, 1e-2),     # Bin 2: low
    (1e-2, 1e-1),     # Bin 3: moderate
    (1e-1, 1.0),      # Bin 4: high
    (1.0, np.inf),    # Bin 5: odorless (>= 1 ppm)
]

BIN_CENTERS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # Representative values per bin


def label_to_threshold(label: float) -> float:
    """
    Convert a predicted (continuous) label to a physical olfactory
    threshold in ppm.

    Args:
        label: Model output (bin index, may be fractional).

    Returns:
        Threshold in ppm (float).
    """
    return 10.0 ** (label - 5)


def label_to_odor_class(label: float) -> bool:
    """
    Map a predicted label to a binary odor class.

    Args:
        label: Model output.

    Returns:
        True if odorous (label < 5, i.e. threshold < 1 ppm),
        False if odorless.
    """
    return label < 5.0


class OdorPredictor:
    """
    Predict odor properties (odorous/odorless and olfactory threshold)
    from molecular structure (SMILES).
    """

    def __init__(self, model_dir="models"):
        """
        Initialize the predictor with a pre-trained model.

        Args:
            model_dir: Directory containing the model file
                       ('best_model.joblib').
        """
        self.model_dir = model_dir
        self.regressor = self._load_model("best_model.joblib")

    def _load_model(self, filename):
        """Load a joblib model file."""
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `python scripts/train_model.py` to train it, or "
                "download the pre-trained model."
            )
        return joblib.load(path)

    def predict(self, smiles: str):
        """
        Predict odor properties for a single molecule.

        Args:
            smiles: SMILES string.

        Returns:
            dict with keys:
                - smiles: input SMILES
                - is_odorous: bool, whether the molecule is predicted odorous
                - threshold_ppm: float, predicted olfactory threshold
                  (only meaningful when is_odorous is True)
                - threshold_bin: int, predicted bin index (0-5)
        """
        fp = smiles_to_morgan(smiles)
        if fp is None:
            return {"smiles": smiles, "is_odorous": None,
                    "threshold_ppm": None, "threshold_bin": None}

        fp_array = np.array(fp).reshape(1, -1)

        raw = float(self.regressor.predict(fp_array)[0])
        bin_idx = int(np.clip(np.round(raw), 0, 5))
        is_odorous = label_to_odor_class(raw)

        return {
            "smiles": smiles,
            "is_odorous": is_odorous,
            "threshold_ppm": label_to_threshold(raw),
            "threshold_bin": bin_idx,
        }

    def predict_from_csv(self, csv_path: str, smiles_column="SMILES"):
        """
        Predict odor properties for molecules listed in a CSV file.

        Args:
            csv_path: Path to CSV file.
            smiles_column: Column name containing SMILES strings.

        Returns:
            pandas DataFrame with prediction results.
        """
        data = pd.read_csv(csv_path)
        results = []

        for smiles in data[smiles_column]:
            result = self.predict(smiles)
            results.append(result)

        return pd.DataFrame(results)
