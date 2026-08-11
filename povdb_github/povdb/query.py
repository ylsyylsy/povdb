"""
POVDB database query module.

Search the Potential Odorous Virtual Database (POVDB) by SMILES,
formula, name or mass.  The POVDB is stored as an MSP spectral library
where each record contains metadata (NAME, CID, FORMULA, INCHIKEY,
SMILES, ...) and a virtual EI-MS spectrum (m/z intensity pairs).
"""

import os
import numpy as np
from .utils import (parse_msp_to_dicts, smiles_to_maccs,
                    tanimoto_similarity)


class POVDBQuery:
    """
    Query the Potential Odorous Virtual Database (POVDB).

    The full POVDB contains 737,519 molecules predicted to be odorous,
    each with a virtual EI-MS spectrum for non-targeted analysis.
    A small sample library is shipped in ``data/sample_povdb.msp``;
    the complete library is available from the corresponding authors.
    """

    def __init__(self, db_path=None):
        """
        Initialize the POVDB query interface.

        Args:
            db_path: Path to the POVDB MSP file. If None, tries
                     'data/sample_povdb.msp' relative to the repo root.
        """
        if db_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(repo_root, "data", "sample_povdb.msp")
        self.db_path = db_path
        self.records = parse_msp_to_dicts(db_path) if os.path.exists(db_path) else []
        print(f"POVDB loaded: {len(self.records)} records from {db_path}")

    def query_by_smiles(self, smiles: str, exact: bool = True):
        """
        Search POVDB by SMILES.

        Args:
            smiles: SMILES string to search.
            exact: If True, match the standardized SMILES exactly;
                   otherwise use MACCS Tanimoto similarity.

        Returns:
            List of matching records (dicts), or [] if none.
        """
        from .utils import standardize_smiles
        target = standardize_smiles(smiles)
        if target is None:
            return []
        results = []
        for rec in self.records:
            rec_smi = rec.get("SMILES", "")
            if exact:
                std = standardize_smiles(rec_smi)
                if std == target:
                    results.append(rec)
            else:
                sim = self._smiles_similarity(smiles, rec_smi)
                if sim >= 0.6:
                    rec["_tanimoto"] = round(sim, 4)
                    results.append(rec)
        return results

    def query_by_formula(self, formula: str):
        """
        Search POVDB by molecular formula (exact match).

        Args:
            formula: Molecular formula (e.g., "C3H6O").

        Returns:
            List of matching records.
        """
        formula = formula.strip()
        return [r for r in self.records if r.get("FORMULA", "").strip() == formula]

    def query_by_name(self, name: str):
        """
        Search POVDB by compound name (case-insensitive substring).

        Args:
            name: Compound name (e.g., "acetone").

        Returns:
            List of matching records.
        """
        name_l = name.strip().lower()
        return [r for r in self.records
                if name_l in r.get("NAME", "").lower()]

    def query_by_mass(self, mz: float, tolerance: float = 0.01):
        """
        Search POVDB by neutral exact mass.

        Args:
            mz: Target exact mass (Da).
            tolerance: Absolute mass tolerance (Da).

        Returns:
            List of matching records.
        """
        results = []
        for rec in self.records:
            mass = rec.get("EXACTMASS", "NA")
            try:
                if abs(float(mass) - mz) <= tolerance:
                    results.append(rec)
            except (ValueError, TypeError):
                continue
        return results

    def find_similar(self, smiles: str, threshold: float = 0.6):
        """
        Find molecules in POVDB with MACCS fingerprint Tanimoto
        similarity above the given threshold (the applicability domain
        criterion used in the manuscript).

        Args:
            smiles: Query SMILES.
            threshold: Minimum Tanimoto similarity (default: 0.6).

        Returns:
            List of (record, similarity) tuples, sorted descending.
        """
        query_fp = smiles_to_maccs(smiles)
        if query_fp is None:
            return []

        matches = []
        for rec in self.records:
            rec_fp = smiles_to_maccs(rec.get("SMILES", ""))
            if rec_fp is None:
                continue
            sim = tanimoto_similarity(query_fp, rec_fp)
            if sim >= threshold:
                matches.append((rec, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _smiles_similarity(self, smiles_a: str, smiles_b: str) -> float:
        fp_a = smiles_to_maccs(smiles_a)
        fp_b = smiles_to_maccs(smiles_b)
        if fp_a is None or fp_b is None:
            return 0.0
        return tanimoto_similarity(fp_a, fp_b)

    @staticmethod
    def _parse_peaks(peak_lines):
        """Parse MSP peak lines ('mz intensity') into (mz, intensity) tuples."""
        peaks = []
        for line in peak_lines:
            parts = line.replace('\t', ' ').split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        return peaks

    def search_spectrum(self, query_peaks, tolerance: float = 0.3,
                        min_peaks: int = 5, top_k: int = 10):
        """
        Match an experimental mass spectrum against POVDB virtual spectra.

        Args:
            query_peaks: List of (mz, intensity) tuples or an MSP peak
                         string list.
            tolerance: m/z tolerance for peak matching (Da).
            min_peaks: Minimum number of query peaks a record must match.
            top_k: Maximum number of hits to return.

        Returns:
            List of (record, matched_peaks, score) sorted by score.
        """
        if query_peaks and isinstance(query_peaks[0], str):
            q_peaks = []
            for line in query_peaks:
                parts = line.replace('\t', ' ').split()
                if len(parts) >= 2:
                    q_peaks.append((float(parts[0]), float(parts[1])))
        else:
            q_peaks = [(float(m), float(i)) for m, i in query_peaks]

        q_mz = np.array([p[0] for p in q_peaks])
        q_int = np.array([p[1] for p in q_peaks])

        hits = []
        for rec in self.records:
            lib_peaks = self._parse_peaks(rec.get("PEAKS", []))
            if len(lib_peaks) == 0:
                continue
            lib_mz = np.array([p[0] for p in lib_peaks])
            lib_int = np.array([p[1] for p in lib_peaks])

            # match query peaks to library peaks within tolerance
            matched = 0
            score = 0.0
            for qm, qi in zip(q_mz, q_int):
                dist = np.abs(lib_mz - qm)
                idx = int(np.argmin(dist))
                if dist[idx] <= tolerance:
                    matched += 1
                    score += qi * lib_int[idx]
            if matched >= min_peaks:
                hits.append((rec, matched, score))

        hits.sort(key=lambda x: x[2], reverse=True)
        return hits[:top_k]
