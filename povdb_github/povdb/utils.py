"""
Utility functions for molecular fingerprint generation, SMILES
standardization, and MSP spectral library I/O.
"""

import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np


# ---------------------------------------------------------------------------
# Molecular fingerprints
# ---------------------------------------------------------------------------

def smiles_to_maccs(smiles: str):
    """
    Convert a SMILES string to a MACCS fingerprint bit vector.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        List of 167 integers representing the MACCS fingerprint,
        or None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return [int(b) for b in fp.ToBitString()]
    except Exception:
        return None


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Convert a SMILES string to a Morgan (circular) fingerprint.

    Args:
        smiles: SMILES string of the molecule.
        radius: Morgan fingerprint radius (default: 2).
        n_bits: Number of bits in the fingerprint (default: 2048).

    Returns:
        List of integers representing the Morgan fingerprint,
        or None if the SMILES is invalid.
    """
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


def tanimoto_similarity(fp1, fp2):
    """
    Compute Tanimoto similarity between two binary fingerprint vectors.

    Args:
        fp1, fp2: Lists of binary integers (0 or 1).

    Returns:
        Tanimoto similarity score (float between 0 and 1).
    """
    arr1 = np.array(fp1, dtype=bool)
    arr2 = np.array(fp2, dtype=bool)
    intersection = np.sum(arr1 & arr2)
    union = np.sum(arr1 | arr2)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# SMILES standardization
# ---------------------------------------------------------------------------

def standardize_smiles(smi: str, isomeric: bool = False):
    """
    Standardize a SMILES string following a PubChem-like protocol:
    keep the largest fragment, neutralize charges, canonicalize tautomers.

    Args:
        smi: Input SMILES string.
        isomeric: Whether to keep stereochemical information.

    Returns:
        Standardized SMILES, or None on failure.
    """
    if not smi or not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def has_isotope(smiles: str) -> bool:
    """
    Detect isotope-labelled atoms in a SMILES string
    (e.g. '[13C]', '[2H]', '[D]').

    Args:
        smiles: SMILES string.

    Returns:
        True if the SMILES contains an isotope marker.
    """
    if not isinstance(smiles, str):
        return False
    return bool(re.search(r'\[(\d|D)', smiles))


# ---------------------------------------------------------------------------
# MSP spectral library I/O
# ---------------------------------------------------------------------------

def parse_msp_to_dicts(file_path: str) -> list:
    """
    Parse an MSP file into a list of record dictionaries.

    Each record dictionary contains the metadata fields (uppercased keys)
    plus a 'PEAKS' list of 'mz intensity' lines.

    Args:
        file_path: Path to the MSP file.

    Returns:
        List of record dicts.
    """
    if not os.path.exists(file_path):
        print(f"Warning: file not found -> {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    records_str = content.strip().split('\n\n')
    parsed_records = []

    for record_data in records_str:
        if not record_data.strip():
            continue

        record_dict = {}
        peaks = []
        lines = record_data.split('\n')

        num_peaks_line_found = False
        for line in lines:
            if num_peaks_line_found:
                if '\t' in line or ' ' in line:
                    peaks.append(line)
            elif ':' in line:
                key, value = line.split(':', 1)
                record_dict[key.strip().upper()] = value.strip()
                if key.strip().upper() == 'NUM PEAKS':
                    num_peaks_line_found = True

        record_dict['PEAKS'] = peaks
        parsed_records.append(record_dict)

    return parsed_records


def records_to_msp_string(records: list) -> str:
    """
    Convert a list of record dictionaries back to an MSP-format string.

    Args:
        records: List of record dicts (as produced by parse_msp_to_dicts).

    Returns:
        MSP-format string.
    """
    msp_strings = []
    order = ['NAME', 'CID', 'EXACTMASS', 'FORMULA', 'ONTOLOGY', 'INCHIKEY',
             'SMILES', 'RETENTIONTIMEINDEX', 'IONMODE', 'INSTRUMENTTYPE',
             'COMMENT']

    for record in records:
        record_str = []
        for key in order:
            if key in record:
                record_str.append(f"{key.capitalize()}: {record[key]}")

        for key, value in record.items():
            if key.upper() not in order and key.upper() != 'PEAKS':
                record_str.append(f"{key.capitalize()}: {value}")

        peaks = record.get('PEAKS', [])
        record_str.append(f"Num Peaks: {len(peaks)}")
        record_str.extend(peaks)
        msp_strings.append('\n'.join(record_str))

    return '\n\n'.join(msp_strings) + '\n'


def remove_isotope_records(records: list) -> list:
    """
    Remove records whose SMILES contains an isotope marker.

    Args:
        records: List of record dicts.

    Returns:
        Filtered list of record dicts.
    """
    final_records = []
    for record in records:
        smi = record.get('SMILES', '')
        if isinstance(smi, str) and has_isotope(smi):
            continue
        final_records.append(record)
    return final_records
