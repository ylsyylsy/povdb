"""
Peak annotator for non-targeted analysis data.

Maps GC-QTOF/MS-DIAL peak tables to the POVDB for identifying
odorous compounds and calculating odor contributions.

Workflow:
    1. Match each detected peak to POVDB entries (by name / formula /
       spectral similarity).
    2. Assign odor properties (odorous/odorless, predicted threshold).
    3. Calculate the odor contribution of each compound as
       (peak area) / (olfactory threshold), and the total contribution.
"""

import os
import numpy as np
import pandas as pd
from .predictor import OdorPredictor
from .utils import standardize_smiles


class PeakAnnotator:
    """
    Annotate non-targeted analysis peaks with POVDB odor properties.
    """

    def __init__(self, povdb_path=None, model_dir="models", predictor=None):
        """
        Initialize the annotator.

        Args:
            povdb_path: Path to the POVDB spectral library (MSP file).
            model_dir: Directory containing the pre-trained model.
            predictor: Optional OdorPredictor instance (avoids
                       re-loading the model).
        """
        from .query import POVDBQuery
        # POVDBQuery(None) falls back to data/sample_povdb.msp automatically
        self.povdb = POVDBQuery(povdb_path)
        self.predictor = predictor or OdorPredictor(model_dir=model_dir)

    def annotate(self, peak_table_path: str, smiles_column="SMILES",
                 area_column="Peak Area"):
        """
        Annotate a peak table from non-targeted analysis.

        Expected input: MS-DIAL export (tab-separated) or CSV with
        columns including 'SMILES' (or 'Name'), 'Peak Area', etc.

        Args:
            peak_table_path: Path to the peak table file.
            smiles_column: Column containing SMILES strings.
            area_column: Column containing peak areas.

        Returns:
            pandas DataFrame with added odor annotation columns:
            Is_Odorous, Threshold_ppm, Odor_Contribution.
        """
        # Choose delimiter by file extension, then verify it parses
        # into multiple columns correctly
        ext = os.path.splitext(peak_table_path)[1].lower()
        candidate_seps = [',', '\t', ';'] if ext == '.csv' else \
                         ['\t', ',', ';']
        data = None
        for sep in candidate_seps:
            try:
                probe = pd.read_csv(peak_table_path, sep=sep,
                                    encoding='utf-8-sig', nrows=3)
                if len(probe.columns) > 1:
                    data = pd.read_csv(peak_table_path, sep=sep,
                                       encoding='utf-8-sig')
                    break
            except Exception:
                continue
        if data is None:
            raise ValueError(f"Could not read peak table: {peak_table_path}")

        print(f"Loaded {len(data)} peaks from {peak_table_path}")

        # Add annotation columns
        data["Is_Odorous"] = None
        data["Threshold_ppm"] = None
        data["Odor_Contribution"] = None

        smiles_source = smiles_column if smiles_column in data.columns else None
        name_source = "Name" if "Name" in data.columns else None

        for idx, row in data.iterrows():
            smiles = row.get(smiles_source) if smiles_source else None
            if isinstance(smiles, str) and smiles.strip():
                pred = self.predictor.predict(smiles.strip())
                data.at[idx, "Is_Odorous"] = pred["is_odorous"]
                data.at[idx, "Threshold_ppm"] = pred["threshold_ppm"]
            elif name_source:
                # fall back to name-based POVDB lookup
                name = row.get(name_source)
                if self.povdb is not None and isinstance(name, str):
                    hits = self.povdb.query_by_name(name)
                    if hits:
                        rec = hits[0]
                        smi = rec.get("SMILES")
                        if smi and smi != "NA":
                            pred = self.predictor.predict(smi)
                            data.at[idx, "Is_Odorous"] = pred["is_odorous"]
                            data.at[idx, "Threshold_ppm"] = pred["threshold_ppm"]

        if area_column in data.columns:
            data = self.calculate_odor_contribution(data, area_column)

        return data

    def calculate_odor_contribution(self, annotated_data: pd.DataFrame,
                                    area_column="Peak Area"):
        """
        Calculate odor contribution for each annotated compound.

        Odor contribution = (Compound Peak Area) / (Olfactory Threshold)

        Args:
            annotated_data: DataFrame from annotate().
            area_column: Column containing peak areas.

        Returns:
            DataFrame with 'Odor_Contribution' column populated.
        """
        if "Threshold_ppm" not in annotated_data.columns:
            raise ValueError("Missing 'Threshold_ppm' column. Run annotate() first.")

        if area_column not in annotated_data.columns:
            print(f"Warning: column '{area_column}' not found; "
                  "odor contribution not computed.")
            return annotated_data

        mask = (annotated_data["Threshold_ppm"].notna()
                & (annotated_data["Threshold_ppm"] > 0)
                & (annotated_data[area_column].notna()))
        annotated_data.loc[mask, "Odor_Contribution"] = (
            annotated_data.loc[mask, area_column]
            / annotated_data.loc[mask, "Threshold_ppm"]
        )

        # Normalize to fraction of total odor contribution
        total = annotated_data["Odor_Contribution"].sum()
        if total > 0:
            annotated_data["Odor_Contribution_Norm"] = (
                annotated_data["Odor_Contribution"] / total
            )
        else:
            annotated_data["Odor_Contribution_Norm"] = 0.0

        return annotated_data

    def summarize(self, annotated_data: pd.DataFrame):
        """
        Generate a summary of odor annotation results.

        Args:
            annotated_data: DataFrame from annotate().

        Returns:
            dict of summary statistics.
        """
        n_total = len(annotated_data)
        n_odorous = int(annotated_data["Is_Odorous"].sum()) \
            if "Is_Odorous" in annotated_data.columns else 0

        summary = {
            "total_peaks": n_total,
            "odorous_compounds": n_odorous,
        }
        if "Odor_Contribution" in annotated_data.columns:
            summary["total_odor_contribution"] = float(
                annotated_data["Odor_Contribution"].sum()
            )
        return summary
