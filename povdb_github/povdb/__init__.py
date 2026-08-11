"""
POVDB: Potential Odorous Virtual Database.

A machine learning-enhanced olfactory data hub for odor-free urban management.

This package provides the core tools described in:

    Wang et al., "Embedding an Olfactory Data Hub in Smart Cities for
    Odor-Free Urban Management", Science Advances (2026).

Modules:
    utils       - Molecular fingerprints, SMILES standardization, MSP I/O
    predictor   - Two-step odor property prediction (odorous/odorless + threshold)
    query       - Search the POVDB spectral library
    annotator   - Annotate GC-QTOF/MS-DIAL peaks with POVDB
"""

__version__ = "1.0.0"

# Silence RDKit deprecation/log noise from fingerprint generation
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from .predictor import OdorPredictor
from .query import POVDBQuery
from .annotator import PeakAnnotator

__all__ = ["OdorPredictor", "POVDBQuery", "PeakAnnotator", "__version__"]
