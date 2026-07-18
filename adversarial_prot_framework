#!/usr/bin/env python3
"""
================================================================================
MULTI-LEVEL ADVERSARIAL PROTEIN EDITING FRAMEWORK v4.0 — PRODUCTION REFACTOR
================================================================================

A tiered protein sequence analysis and conservative editing pipeline.
Each level adds increasingly stringent biophysical constraints. Proteins
are tagged with a "level certificate" indicating which tiers they passed.

LEVEL HIERARCHY
---------------
Level 1: Sequence Validity — Basic IUPAC alphabet and length checks
Level 2: Physicochemical Compatibility — Substitution scoring, hydropathy, 
         secondary structure propensity, aggregation propensity, burial
Level 3: Structural Integrity — Fold confidence, allosteric networks,
         conformational dynamics, secondary structure transitions
Level 4: Evolutionary Conservation — Co-evolutionary coupling, phylogenetic
         conservation scores
Level 5: Functional/Phenotypic — Catalytic site preservation, binding
         interface disruption

USAGE
-----
    # Score a mutation
    python protein_editor.py --wt "MKTLLIL..." --mut "MKTLLIL..." --levels 1 2 3

    # Suggest conservative mutations
    python protein_editor.py --wt "MKTLLIL..." --suggest 5 --strategy conservative

    # Batch process a FASTA database
    python protein_editor.py --input proteins.fasta --output results.json --levels 1 2 3 4 5

AUTHOR: Refactored from v3.0
DATE: 2026-07-18
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# SECTION 0: CONFIGURATION & CONSTANTS
# ==============================================================================

# Standard 20 amino acids (IUPAC)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AMINO_ACIDS)
AA_LIST = list(AMINO_ACIDS)
AA_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# ------------------------------------------------------------------------------
# 0.1 KYTE-DOOLITTLE HYDROPATHY SCALE
# Source: Kyte J., Doolittle R.F. (1982) J. Mol. Biol. 157:105-132
# DOI: 10.1016/0022-2836(82)90515-0
# What it checks: Hydrophobic/hydrophilic character of amino acid side chains
# Evaluation: Mean absolute difference between WT and mutant hydropathy values,
#             normalized by the scale range (~9.0)
# ------------------------------------------------------------------------------
HYDROPATHY = np.array([
    1.80,   # A - Alanine
    2.50,   # C - Cysteine
    -3.50,  # D - Aspartic acid
    -3.50,  # E - Glutamic acid
    2.80,   # F - Phenylalanine
    -0.40,  # G - Glycine
    -3.20,  # H - Histidine
    4.50,   # I - Isoleucine
    -3.90,  # K - Lysine
    3.80,   # L - Leucine
    1.90,   # M - Methionine
    -3.50,  # N - Asparagine
    -1.60,  # P - Proline
    -3.50,  # Q - Glutamine
    -4.50,  # R - Arginine
    -0.80,  # S - Serine
    -0.70,  # T - Threonine
    4.20,   # V - Valine
    -0.90,  # W - Tryptophan
    -1.30,  # Y - Tyrosine
], dtype=np.float32)

HYDROPATHY_RANGE = float(np.max(HYDROPATHY) - np.min(HYDROPATHY))  # ~9.0

# ------------------------------------------------------------------------------
# 0.2 GRANTHAM CHEMICAL DISTANCE MATRIX
# Source: Grantham R. (1974) Science 185:862-864
# DOI: 10.1126/science.185.4154.862
# What it checks: Composite chemical distance based on composition, polarity,
#                 and molecular volume of amino acid side chains
# Evaluation: Mean Grantham distance between WT->mut substitutions, normalized
#             by the maximum possible distance (215.0)
# ------------------------------------------------------------------------------
GRANTHAM = np.array([
    [0,  195, 126, 107, 113, 60,  86,  94,  106, 96,  84,  111, 27,  91,  112, 99,  58,  64,  148, 112],
    [195, 0,  154, 170, 205, 159, 174, 198, 202, 198, 196, 139, 169, 154, 180, 112, 81,  192, 215, 194],
    [126, 154, 0,   45,  177, 94,  101, 168, 101, 172, 160, 23,  108, 61,  96,  65,  85,  152, 181, 160],
    [107, 170, 45,  0,   140, 98,  83,  134, 56,  138, 126, 42,  93,  29,  54,  80,  65,  121, 152, 122],
    [113, 205, 177, 140, 0,   153, 100, 21,  102, 22,  28,  158, 114, 116, 97,  155, 134, 50,  40,  22],
    [60,  159, 94,  98,  153, 0,   98,  135, 127, 138, 127, 80,  42,  87,  125, 56,  59,  109, 184, 147],
    [86,  174, 101, 83,  100, 98,  0,   94,  32,  99,  87,  68,  77,  24,  29,  81,  47,  84,  115, 83],
    [94,  198, 168, 134, 21,  135, 94,  0,   102, 5,   10,  149, 95,  109, 97,  142, 89,  29,  61,  33],
    [106, 202, 101, 56,  102, 127, 32,  102, 0,   107, 95,  94,  103, 53,  26,  121, 78,  97,  110, 85],
    [96,  198, 172, 138, 22,  138, 99,  5,   107, 0,   15,  153, 98,  113, 102, 145, 92,  32,  61,  36],
    [84,  196, 160, 126, 28,  127, 87,  10,  95,  15,  0,   142, 87,  101, 91,  135, 81,  21,  67,  36],
    [111, 139, 23,  42,  158, 80,  68,  149, 94,  153, 142, 0,   91,  46,  86,  46,  65,  133, 174, 143],
    [27,  169, 108, 93,  114, 42,  77,  95,  103, 98,  87,  91,  0,   76,  103, 74,  38,  68,  147, 110],
    [91,  154, 61,  29,  116, 87,  24,  109, 53,  113, 101, 46,  76,  0,   43,  68,  42,  96,  130, 99],
    [112, 180, 96,  54,  97,  125, 29,  97,  26,  102, 91,  86,  103, 43,  0,   110, 71,  96,  101, 77],
    [99,  112, 65,  80,  155, 56,  81,  142, 121, 145, 135, 46,  74,  68,  110, 0,   58,  124, 177, 144],
    [58,  81,  85,  65,  134, 59,  47,  89,  78,  92,  81,  65,  38,  42,  71,  58,  0,   69,  128, 92],
    [64,  192, 152, 121, 50,  109, 84,  29,  97,  32,  21,  133, 68,  96,  96,  124, 69,  0,   88,  55],
    [148, 215, 181, 152, 40,  184, 115, 61,  110, 61,  67,  174, 147, 130, 101, 177, 128, 88,  0,   37],
    [112, 194, 160, 122, 22,  147, 83,  33,  85,  36,  36,  143, 110, 99,  77,  144, 92,  55,  37,  0],
], dtype=np.float32)
GRANTHAM_MAX = 215.0

# ------------------------------------------------------------------------------
# 0.3 BLOSUM62 SUBSTITUTION MATRIX
# Source: Henikoff S., Henikoff J.G. (1992) PNAS 89:10915-10919
# DOI: 10.1073/pnas.89.22.10915
# What it checks: Log-odds of amino acid substitutions based on observed
#                 frequencies in conserved protein blocks (BLOCKS database)
# Evaluation: Mean BLOSUM62 score of substitutions. Positive = conservative,
#             negative = radical. Normalized to [0,1] penalty scale.
# ------------------------------------------------------------------------------
BLOSUM62 = np.array([
    [4,  -1, -2, -2, 0,  -1, -1, 0,  -2, -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],
    [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3],
    [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3],
    [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2],
    [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2],
    [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3],
    [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -3, 2,  0,  -3, -2, -1, -2, -1, 1],
    [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -3, 5,  -1, -3, -1, 0,  -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2],
    [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2],
    [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1],
    [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],
], dtype=np.float32)

# ------------------------------------------------------------------------------
# 0.4 CHOU-FASMAN SECONDARY STRUCTURE PROPENSITIES
# Source: Chou P.Y., Fasman G.D. (1974) Biochemistry 13:222-245
# DOI: 10.1021/bi00699a002
# What it checks: Intrinsic propensity of each amino acid to form alpha-helices,
#                 beta-sheets, or turns (coils)
# Evaluation: Euclidean distance between WT and mutant propensity vectors
#             (3D: [P_alpha, P_beta, P_turn]), normalized by ~2.0
# ------------------------------------------------------------------------------
# Columns: [P(alpha-helix), P(beta-sheet), P(turn)]
CHOU_FASMAN = np.array([
    [1.45, 0.97, 0.66],  # A
    [0.77, 1.30, 0.54],  # C
    [0.98, 0.80, 1.46],  # D
    [1.53, 0.26, 0.74],  # E
    [1.12, 1.28, 0.60],  # F
    [0.53, 0.81, 1.56],  # G
    [1.24, 0.71, 0.95],  # H
    [1.00, 1.60, 0.47],  # I
    [1.07, 0.74, 1.01],  # K
    [1.34, 1.22, 0.57],  # L
    [1.20, 1.67, 0.47],  # M
    [0.73, 0.65, 1.49],  # N
    [0.59, 0.62, 1.52],  # P
    [1.17, 1.23, 0.75],  # Q
    [0.79, 0.90, 0.93],  # R
    [0.75, 0.72, 1.43],  # S
    [0.82, 1.20, 0.96],  # T
    [0.91, 1.87, 0.41],  # V
    [1.14, 1.19, 0.75],  # W
    [0.76, 1.45, 0.76],  # Y
], dtype=np.float32)

# ------------------------------------------------------------------------------
# 0.5 AGGREGATION PROPENSITY (TANGO-like scale)
# Source: Derived from aggregation-prone sequence analysis
#         Fernandez-Escamilla et al. (2004) Nature Biotechnology 22:1302
#         Pawar et al. (2005) Journal of Molecular Biology 350:379
# What it checks: Intrinsic tendency of amino acids to promote protein
#                 aggregation (amyloid-like or amorphous)
# Evaluation: Mean change in aggregation propensity (mut - wt). Positive
#             values indicate increased aggregation risk.
# ------------------------------------------------------------------------------
AGGREGATION = np.array([
    0.30,  # A
    0.50,  # C
    0.10,  # D
    0.10,  # E
    0.80,  # F
    0.20,  # G
    0.40,  # H
    0.90,  # I
    0.10,  # K
    0.80,  # L
    0.60,  # M
    0.20,  # N
    0.00,  # P
    0.30,  # Q
    0.10,  # R
    0.20,  # S
    0.30,  # T
    0.80,  # V
    0.70,  # W
    0.70,  # Y
], dtype=np.float32)

# ------------------------------------------------------------------------------
# 0.6 MAXIMUM SOLVENT ACCESSIBLE SURFACE AREA (Angstrom^2)
# Source: Tien et al. (2013) PLoS ONE 8(2): e80635
# What it checks: Expected surface exposure of each amino acid type
# Evaluation: Relative solvent accessibility (RSA) = SASA / MAX_ASA
# ------------------------------------------------------------------------------
MAX_ASA = np.array([
    106,  # A
    135,  # C
    163,  # D
    194,  # E
    197,  # F
    84,   # G
    184,  # H
    169,  # I
    205,  # K
    164,  # L
    188,  # M
    157,  # N
    136,  # P
    189,  # Q
    248,  # R
    130,  # S
    142,  # T
    142,  # V
    227,  # W
    222,  # Y
], dtype=np.float32)

# ------------------------------------------------------------------------------
# 0.7 RESIDUE CLASSIFICATIONS
# ------------------------------------------------------------------------------
HYDROPHOBIC = frozenset("AILMFWVY")
POLAR_CHARGED = frozenset("KRHDEQNSTY")
CATALYTIC = frozenset("HDECKSY")  # Common catalytic residues

# ------------------------------------------------------------------------------
# 0.8 DSSP 8-STATE -> 3-STATE MAPPING
# Source: Kabsch W., Sander C. (1983) Biopolymers 22:2577-2637
# What it checks: Secondary structure classification from 3D coordinates
# ------------------------------------------------------------------------------
DSSP_3STATE = {
    "H": "H", "G": "H", "I": "H",   # Helix (alpha, 3_10, pi)
    "E": "E", "B": "E",             # Sheet (extended, bridge)
    "T": "C", "S": "C", "C": "C", "-": "C",  # Coil/turn
}

# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES
# ==============================================================================

class CombinationMode(Enum):
    """How multiple constraint scores are combined into a panel score."""
    ADDITIVE = "additive"           # Weighted arithmetic mean
    MULTIPLICATIVE = "multiplicative"  # Weighted geometric mean
    HARD_GATES = "hard_gates"       # Maximum (most permissive)
    HYBRID = "hybrid"               # Simple arithmetic mean (unweighted)


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Result of evaluating a single constraint."""
    name: str
    score: float                    # [0, 1] where 0 = perfect, 1 = worst
    applicable: bool                # Was this constraint applicable?
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class PanelResult:
    """Result of evaluating a full constraint panel (one level)."""
    constraint_results: Tuple[ConstraintResult, ...]
    aggregate_score: float          # Combined score [0, 1]
    n_applicable: int
    n_total: int
    combination_mode: CombinationMode
    passed: bool                    # aggregate_score < threshold?
    applicable_weights_sum: float


@dataclass(frozen=True, slots=True)
class LevelReport:
    """Result of evaluating one level in the hierarchy."""
    level: int
    panel_result: PanelResult
    passed: bool
    time_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class LevelCertificate:
    """
    Immutable certificate proving which levels a protein edit passed.

    This is the key tagging mechanism. Every evaluated protein gets a
    certificate that records exactly which levels it cleared.

    Example:
        cert = LevelCertificate(passed_levels=(1, 2, 3), 
                                stopped_at=3, 
                                final_score=0.23)
        cert.highest_level  # -> 3
        cert.has_passed(2)  # -> True
        cert.tag          # -> "L3" (or "L1-L3" for full range)
    """
    passed_levels: Tuple[int, ...]
    stopped_at_level: int
    final_score: float
    final_passed: bool
    level_scores: Dict[int, float] = field(default_factory=dict, compare=False)

    @property
    def highest_level(self) -> int:
        return max(self.passed_levels) if self.passed_levels else 0

    @property
    def tag(self) -> str:
        """Short tag: e.g., 'L3' or 'L1-L3' or 'FAIL-L1'."""
        if not self.passed_levels:
            return f"FAIL-L{self.stopped_at_level}"
        if len(self.passed_levels) == 1:
            return f"L{self.passed_levels[0]}"
        return f"L{self.passed_levels[0]}-L{self.passed_levels[-1]}"

    @property
    def full_tag(self) -> str:
        """Detailed tag with score info."""
        levels = ",".join(f"L{l}:{self.level_scores.get(l, 'N/A'):.3f}" 
                         for l in self.passed_levels)
        return f"[{self.tag}] scores=({levels}) final={self.final_score:.3f}"

    def has_passed(self, level: int) -> bool:
        return level in self.passed_levels

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "full_tag": self.full_tag,
            "passed_levels": list(self.passed_levels),
            "highest_level": self.highest_level,
            "stopped_at_level": self.stopped_at_level,
            "final_score": round(self.final_score, 4),
            "final_passed": self.final_passed,
            "level_scores": {k: round(v, 4) for k, v in self.level_scores.items()},
        }


@dataclass(frozen=True, slots=True)
class MultiLevelReport:
    """Complete report for a WT->mut evaluation."""
    wt_seq: str
    mut_seq: str
    n_mutations: int
    level_reports: Tuple[LevelReport, ...]
    final_score: float
    final_passed: bool
    stopped_at_level: int
    certificate: LevelCertificate
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wt_seq": self.wt_seq,
            "mut_seq": self.mut_seq,
            "n_mutations": self.n_mutations,
            "certificate": self.certificate.to_dict(),
            "final_score": round(self.final_score, 4),
            "final_passed": self.final_passed,
            "stopped_at_level": self.stopped_at_level,
            "levels": [
                {
                    "level": lr.level,
                    "passed": lr.passed,
                    "score": round(lr.panel_result.aggregate_score, 4),
                    "threshold": "< 0.5",
                    "time_ms": round(lr.time_ms, 2),
                    "constraints": [
                        {
                            "name": cr.name,
                            "score": round(cr.score, 4),
                            "applicable": cr.applicable,
                            "weight": cr.weight,
                            "metadata": cr.metadata,
                        }
                        for cr in lr.panel_result.constraint_results
                    ],
                }
                for lr in self.level_reports
            ],
            "metadata": self.metadata,
        }


# ==============================================================================
# SECTION 2: CONSTRAINT ABSTRACTION
# ==============================================================================

class Constraint(ABC):
    """Abstract base for all biophysical constraints."""

    __slots__ = ("name", "weight", "_cache")

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self._cache: Dict[int, ConstraintResult] = {}

    @abstractmethod
    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        """Evaluate the constraint between wild-type and mutant sequences."""
        pass

    def evaluate_cached(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        """Cached version for repeated evaluations."""
        key = hash((wt_seq, mut_seq, tuple(sorted(kwargs.items()))))
        if key not in self._cache:
            self._cache[key] = self.evaluate(wt_seq, mut_seq, **kwargs)
        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()


# ==============================================================================
# SECTION 3: PANEL (LEVEL) IMPLEMENTATION
# ==============================================================================

class ProtoPanel:
    """A panel groups multiple constraints into one evaluation level."""

    __slots__ = ("constraints", "mode", "threshold", "level_name")

    def __init__(
        self,
        constraints: Sequence[Constraint],
        mode: CombinationMode = CombinationMode.HYBRID,
        threshold: float = 0.5,
        level_name: str = "",
    ):
        self.constraints = tuple(constraints)
        self.mode = mode
        self.threshold = threshold
        self.level_name = level_name

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> PanelResult:
        results: List[ConstraintResult] = []
        app_scores: List[float] = []
        app_weights: List[float] = []

        for c in self.constraints:
            r = c.evaluate_cached(wt_seq, mut_seq, **kwargs)
            results.append(r)
            if r.applicable:
                app_scores.append(r.score)
                app_weights.append(r.weight)

        n_app = len(app_scores)
        n_total = len(results)

        if n_app == 0:
            return PanelResult(
                tuple(results), 0.5, 0, n_total, self.mode, True, 0.0
            )

        scores_arr = np.array(app_scores, dtype=np.float32)
        weights_arr = np.array(app_weights, dtype=np.float32)
        total_w = float(weights_arr.sum())

        if total_w == 0:
            norm = np.ones(n_app) / n_app
        else:
            norm = weights_arr / total_w

        if self.mode == CombinationMode.ADDITIVE:
            agg = float(np.dot(scores_arr, norm))
        elif self.mode == CombinationMode.MULTIPLICATIVE:
            log_scores = np.log(np.maximum(scores_arr, 1e-10))
            agg = float(np.exp(np.dot(log_scores, norm)))
        elif self.mode == CombinationMode.HARD_GATES:
            agg = float(np.max(scores_arr))
        else:  # HYBRID
            agg = float(np.mean(scores_arr))

        # Clamp to [0, 1]
        agg = max(0.0, min(1.0, agg))

        return PanelResult(
            tuple(results),
            agg,
            n_app,
            n_total,
            self.mode,
            agg < self.threshold,
            total_w,
        )

    def clear_caches(self) -> None:
        for c in self.constraints:
            c.clear_cache()


# ==============================================================================
# SECTION 4: LEVEL 1 — SEQUENCE VALIDITY
# ==============================================================================
# Source: IUPAC-IUB Joint Commission on Biochemical Nomenclature
# What it checks: Legal amino acid alphabet, sequence length, no ambiguous
#                 or non-standard residues
# Evaluation: Binary pass/fail. Score = 0 if valid, 1 if invalid.
# Why Level 1: Every downstream analysis depends on a valid sequence.
# ==============================================================================

class SequenceValidity(Constraint):
    """Checks that the sequence contains only valid amino acid letters."""

    def __init__(self):
        super().__init__("sequence_validity", weight=1.0)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        wt_invalid = [c for c in wt_seq if c not in AA_SET]
        mut_invalid = [c for c in mut_seq if c not in AA_SET]

        if wt_invalid or mut_invalid:
            score = 1.0
            meta = {
                "wt_invalid": wt_invalid,
                "mut_invalid": mut_invalid,
                "wt_length": len(wt_seq),
                "mut_length": len(mut_seq),
            }
        else:
            score = 0.0
            meta = {"wt_length": len(wt_seq), "mut_length": len(mut_seq)}

        return ConstraintResult(self.name, score, True, self.weight, meta)


class LengthMatch(Constraint):
    """Ensures WT and mutant have the same length (no indels)."""

    def __init__(self):
        super().__init__("length_match", weight=1.0)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        if len(wt_seq) != len(mut_seq):
            score = 1.0
            meta = {"wt_len": len(wt_seq), "mut_len": len(mut_seq), "diff": abs(len(wt_seq) - len(mut_seq))}
        else:
            score = 0.0
            meta = {"length": len(wt_seq)}
        return ConstraintResult(self.name, score, True, self.weight, meta)


class NoStopCodons(Constraint):
    """Checks for stop codon characters (X, *, etc.) in the mutant."""

    def __init__(self):
        super().__init__("no_stop_codons", weight=1.0)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        stop_chars = set("X*")
        found = [i for i, c in enumerate(mut_seq) if c in stop_chars]
        score = min(len(found) / max(len(mut_seq), 1), 1.0) if found else 0.0
        return ConstraintResult(self.name, score, True, self.weight, {"stop_positions": found})


def build_level1_panel() -> ProtoPanel:
    """Build Level 1: Sequence Validity panel."""
    return ProtoPanel(
        constraints=[SequenceValidity(), LengthMatch(), NoStopCodons()],
        mode=CombinationMode.HARD_GATES,  # Any failure is fatal
        threshold=0.1,  # Very strict — almost any issue fails
        level_name="Level 1: Sequence Validity",
    )


# ==============================================================================
# SECTION 5: LEVEL 2 — PHYSICOCHEMICAL COMPATIBILITY
# ==============================================================================
# These constraints require no external data (structure, MSA, etc.).
# They operate purely on sequence-derived physicochemical properties.
# ==============================================================================

class GranthamDistance(Constraint):
    """
    Grantham chemical distance between WT and mutant amino acids.

    Source: Grantham R. (1974) Science 185:862-864
    DOI: 10.1126/science.185.4154.862

    What it checks: Composite chemical distance based on:
        - Composition (C) — atomic weight ratio of non-C atoms in side chain
        - Polarity (P) — side chain polarity
        - Volume (V) — side chain volume

    Evaluation: d = sqrt[(C1-C2)^2 + (P1-P2)^2 + (V1-V2)^2]
              Score = mean(d) / 215 (max possible distance)
    """

    def __init__(self):
        super().__init__("grantham_distance", weight=0.3)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [(a, b) for a, b in zip(wt_seq, mut_seq) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        ia = np.array([AA_IDX[a] for a, _ in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[b] for _, b in diffs], dtype=np.int32)
        score = float(np.mean(GRANTHAM[ia, ib]) / GRANTHAM_MAX)
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"n_mutations": len(diffs), "mean_distance": float(np.mean(GRANTHAM[ia, ib]))}
        )


class BLOSUM62Score(Constraint):
    """
    BLOSUM62 substitution score for WT->mut changes.

    Source: Henikoff S., Henikoff J.G. (1992) PNAS 89:10915-10919
    DOI: 10.1073/pnas.89.22.10915

    What it checks: Log-odds that a substitution is observed in conserved
                    protein blocks. Positive = frequent/acceptable;
                    Negative = rare/deleterious.

    Evaluation: Score = max(0, 1 - (mean_BLOSUM + 4) / 15)
              Maps BLOSUM range [-4, 11] -> penalty [0, 1]
    """

    def __init__(self):
        super().__init__("blosum62_score", weight=0.3)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [(a, b) for a, b in zip(wt_seq, mut_seq) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        ia = np.array([AA_IDX[a] for a, _ in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[b] for _, b in diffs], dtype=np.int32)
        mean_s = float(np.mean(BLOSUM62[ia, ib]))
        # Map BLOSUM to penalty: high positive BLOSUM -> low penalty
        score = max(0.0, 1.0 - (mean_s + 4.0) / 15.0)
        return ConstraintResult(
            self.name, score, True, self.weight,
            {"mean_blosum": round(mean_s, 3), "n_mutations": len(diffs)}
        )


class HydropathyChange(Constraint):
    """
    Change in Kyte-Doolittle hydropathy upon mutation.

    Source: Kyte J., Doolittle R.F. (1982) J. Mol. Biol. 157:105-132
    DOI: 10.1016/0022-2836(82)90515-0

    What it checks: Whether mutations preserve the hydrophobic/hydrophilic
                    character of each position.

    Evaluation: Score = mean(|delta hydropathy|) / 9.0
              Large hydropathy flips (e.g., I->R) score near 1.0
    """

    def __init__(self):
        super().__init__("hydropathy_change", weight=0.2)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [(a, b) for a, b in zip(wt_seq, mut_seq) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        ia = np.array([AA_IDX[a] for a, _ in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[b] for _, b in diffs], dtype=np.int32)
        score = float(np.mean(np.abs(HYDROPATHY[ia] - HYDROPATHY[ib])) / HYDROPATHY_RANGE)
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"mean_abs_change": round(float(np.mean(np.abs(HYDROPATHY[ia] - HYDROPATHY[ib]))), 3)}
        )


class SecondaryStructurePropensity(Constraint):
    """
    Change in Chou-Fasman secondary structure propensity.

    Source: Chou P.Y., Fasman G.D. (1974) Biochemistry 13:222-245
    DOI: 10.1021/bi00699a002

    What it checks: Whether mutations preserve the intrinsic tendency to form
                    alpha-helices, beta-sheets, or turns.

    Evaluation: Score = mean(Euclidean distance in [P_alpha, P_beta, P_turn] space) / 2.0
    """

    def __init__(self):
        super().__init__("ss_propensity_change", weight=0.2)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [(a, b) for a, b in zip(wt_seq, mut_seq) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        ia = np.array([AA_IDX[a] for a, _ in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[b] for _, b in diffs], dtype=np.int32)
        dists = np.linalg.norm(CHOU_FASMAN[ia] - CHOU_FASMAN[ib], axis=1)
        score = float(np.mean(dists) / 2.0)
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"mean_propensity_shift": round(float(np.mean(dists)), 3)}
        )


class AggregationPropensity(Constraint):
    """
    Change in intrinsic aggregation propensity.

    Source: Fernandez-Escamilla A.M. et al. (2004) Nature Biotechnology 22:1302
    & Pawar A.P. et al. (2005) Journal of Molecular Biology 350:379

    What it checks: Whether mutations increase the risk of protein aggregation
                    (amyloid-like or amorphous).

    Evaluation: Score = max(0, mean(agg_mut - agg_wt))
              Only penalizes increases in aggregation propensity.
    """

    def __init__(self):
        super().__init__("aggregation_propensity", weight=0.3)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [(a, b) for a, b in zip(wt_seq, mut_seq) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        ia = np.array([AA_IDX[a] for a, _ in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[b] for _, b in diffs], dtype=np.int32)
        mean_chg = float(np.mean(AGGREGATION[ib] - AGGREGATION[ia]))
        score = max(0.0, mean_chg)
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"mean_agg_change": round(mean_chg, 3)}
        )


class BurialChange(Constraint):
    """
    Change in predicted solvent accessibility / burial.

    Source: Tien M.Z. et al. (2013) PLoS ONE 8(2): e80635 (MAX_ASA values)

    What it checks: Whether mutations are compatible with the expected
                    solvent exposure of each position.

    Evaluation: If SASA data provided: mean(|RSA_wt - RSA_mut|)
                Otherwise: fallback to hydropathy difference proxy
    """

    def __init__(self):
        super().__init__("burial_change", weight=0.2)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        sasa = kwargs.get("sasa_wt")
        if sasa is not None and len(sasa) >= len(wt_seq):
            rel_wt = np.array([sasa[i] / MAX_ASA[AA_IDX[wt_seq[i]]] for i in diffs])
            rel_mut = np.array([sasa[i] / MAX_ASA[AA_IDX[mut_seq[i]]] for i in diffs])
            score = float(np.mean(np.abs(rel_wt - rel_mut)))
            return ConstraintResult(
                self.name, min(score, 1.0), True, self.weight,
                {"source": "SASA", "mean_rsa_shift": round(score, 3)}
            )

        # Fallback: use hydropathy as proxy for burial
        ia = np.array([AA_IDX[wt_seq[i]] for i in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[mut_seq[i]] for i in diffs], dtype=np.int32)
        score = float(np.mean(np.abs(HYDROPATHY[ia] - HYDROPATHY[ib])) / HYDROPATHY_RANGE)
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"fallback": True, "source": "hydropathy_proxy"}
        )


class ESM2_wt_marginal(Constraint):
    """
    ESM-2 pseudo-log-likelihood difference between WT and mutant.

    Source: Lin Z. et al. (2023) Science 379:1123-1130
    DOI: 10.1126/science.ade2574

    What it checks: Whether the mutant sequence is plausible under the ESM-2
                    language model (trained on millions of protein sequences).

    Evaluation: Likelihood ratio = PLL(wt) - PLL(mut)
              Score = sigmoid(-LR * 0.5)  [higher LR -> higher penalty]

    NOTE: Requires fair-esm package and GPU. Falls back to BLOSUM62 if
          ESM-2 is unavailable.
    """

    __slots__ = ["device", "_model", "_alphabet", "_batch_converter", "_loaded"]

    def __init__(self, model=None, alphabet=None, device: str = "cuda"):
        super().__init__("esm2_wt_marginal", weight=1.0)
        self.device = device
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = None
        self._loaded = model is not None

    def _load_model(self) -> bool:
        if self._loaded:
            return True
        try:
            import esm
            import torch
            self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(
                "esm2_t33_650M_UR50D"
            )
            self._model = self._model.to(self.device).eval()
            self._batch_converter = self._alphabet.get_batch_converter()
            self._loaded = True
            return True
        except Exception as e:
            warnings.warn(f"ESM-2 loading failed: {e}. Using BLOSUM fallback.")
            self._loaded = True  # Mark as attempted
            return False

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        if not self._load_model():
            return self._fallback(wt_seq, mut_seq)

        try:
            import torch
            wt_ll = self._pll(wt_seq)
            mut_ll = self._pll(mut_seq)
            lr = wt_ll - mut_ll  # positive = mutant is worse
            score = 1.0 / (1.0 + np.exp(-lr * 0.5))
            return ConstraintResult(
                self.name, float(score), True, self.weight,
                {"wt_pll": round(wt_ll, 4), "mut_pll": round(mut_ll, 4), "lr": round(lr, 4)}
            )
        except Exception:
            return self._fallback(wt_seq, mut_seq)

    def _pll(self, seq: str) -> float:
        import torch
        _, _, tokens = self._batch_converter([("protein", seq)])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            logits = self._model(tokens, return_contacts=False)["logits"][0, 1:-1]
        target = tokens[0, 1:-1]
        return float(
            torch.log_softmax(logits, dim=-1)[
                torch.arange(len(target)), target
            ].sum() / len(target)
        )

    def _fallback(self, wt_seq: str, mut_seq: str) -> ConstraintResult:
        diffs = [(a, b) for a, b in zip(wt_seq, mut_seq) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight, {"fallback": True})

        scores = [BLOSUM62[AA_IDX[a], AA_IDX[b]] for a, b in diffs]
        mean_s = float(np.mean(scores))
        score = max(0.0, 1.0 - (mean_s + 4.0) / 15.0)
        return ConstraintResult(
            self.name, float(score), True, self.weight,
            {"fallback": True, "mean_blosum": round(mean_s, 3)}
        )


def build_level2_panel(use_esm2: bool = False, device: str = "cuda") -> ProtoPanel:
    """Build Level 2: Physicochemical Compatibility panel."""
    constraints: List[Constraint] = [
        GranthamDistance(),
        BLOSUM62Score(),
        HydropathyChange(),
        SecondaryStructurePropensity(),
        AggregationPropensity(),
        BurialChange(),
    ]
    if use_esm2:
        constraints.insert(0, ESM2_wt_marginal(device=device))
    return ProtoPanel(
        constraints=constraints,
        mode=CombinationMode.HYBRID,
        threshold=0.5,
        level_name="Level 2: Physicochemical Compatibility",
    )


# ==============================================================================
# SECTION 6: LEVEL 3 — STRUCTURAL INTEGRITY
# ==============================================================================
# These constraints require structural data (pLDDT, DSSP, contact maps,
# RMSF, etc.) but can fall back to sequence-based proxies when unavailable.
# ==============================================================================

class FoldIntegrity(Constraint):
    """
    Structural fold confidence score.

    Source: AlphaFold pLDDT (Jumper J. et al. 2021 Nature 596:583-589)

    What it checks: Whether the mutant maintains a confident structural fold
                    (pLDDT >= 70 is considered reliable).

    Evaluation: If pLDDT provided: score = max(0, (70 - pLDDT) / 70)
                Fallback: sequence complexity penalty (low complexity = bad)
    """

    def __init__(self):
        super().__init__("fold_integrity", weight=1.0)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        plddt = kwargs.get("plddt_produced")
        if plddt is not None:
            score = 0.0 if plddt >= 70 else (70.0 - plddt) / 70.0
            return ConstraintResult(
                self.name, score, True, self.weight,
                {"plddt": plddt, "source": "alphafold"}
            )

        # Fallback: sequence complexity (Shannon entropy of composition)
        counts = np.bincount([AA_IDX[c] for c in mut_seq], minlength=20)
        probs = counts[counts > 0] / len(mut_seq)
        entropy = float(-np.sum(probs * np.log2(probs)))
        complexity = entropy / np.log2(min(20, len(mut_seq)))
        score = max(0.0, 1.0 - complexity)
        return ConstraintResult(
            self.name, score, True, self.weight,
            {"fallback": True, "complexity": round(complexity, 3), "entropy": round(entropy, 3)}
        )


class AllostericNetworkConservation(Constraint):
    """
    Preservation of allosteric communication networks.

    Source: Ribeiro A.A.S.T., Ortiz V. (2015) PNAS 112:6056-6061
    & general allosteric network analysis literature

    What it checks: Whether mutations disrupt known allosteric sites or
                    highly connected network hubs.

    Evaluation: If allosteric sites annotated: penalty per hit
                Fallback: contact density + conservation + SS state proxy
    """

    def __init__(self):
        super().__init__("allosteric_network", weight=0.8)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        allosteric = kwargs.get("allosteric_sites", [])
        if allosteric:
            hits = sum(1 for i in diffs if i in allosteric)
            if hits > 0:
                score = min(hits / len(diffs), 1.0)
                return ConstraintResult(
                    self.name, score, True, self.weight,
                    {"allosteric_hits": hits, "source": "annotated_sites"}
                )

        # Fallback: structural proxy
        contact = kwargs.get("contact_map")
        conservation = kwargs.get("conservation_scores")
        ss = kwargs.get("ss_wt")

        max_pen = 0.0
        for pos in diffs:
            score = 0.0
            if contact is not None and pos < contact.shape[0]:
                score += min(np.sum(contact[pos] > 0.5) / 20.0, 1.0) * 0.3
            if conservation and pos in conservation:
                score += conservation[pos] * 0.3
            if ss and pos < len(ss) and ss[pos] in "HEGI":
                score += 0.25
            if wt_seq[pos] in HYDROPHOBIC:
                score += 0.15
            max_pen = max(max_pen, score)

        return ConstraintResult(
            self.name, min(max_pen, 1.0), True, self.weight,
            {"source": "proxy", "max_penalty": round(max_pen, 3)}
        )


class ConformationalDynamics(Constraint):
    """
    Preservation of conformational flexibility dynamics.

    Source: General MD/RMSF literature (e.g., McCammon J.A. et al. 1977)

    What it checks: Whether mutations alter the flexibility profile of the
                    protein (RMSF changes at mutated positions).

    Evaluation: If RMSF provided: mean(|delta RMSF|) / 5.0
                Fallback: Glycine/Proline flexibility proxy
    """

    def __init__(self):
        super().__init__("conformational_dynamics", weight=0.6)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        rmsf = kwargs.get("rmsf_wt")
        if rmsf is not None:
            changes = [abs(rmsf[i]) for i in diffs if i < len(rmsf)]
            if changes:
                score = float(np.mean(changes) / 5.0)
                return ConstraintResult(
                    self.name, min(score, 1.0), True, self.weight,
                    {"source": "RMSF", "mean_change": round(float(np.mean(changes)), 3)}
                )

        # Fallback: Glycine (flexible) and Proline (rigid) as proxies
        changes = []
        for i in diffs:
            wt_f = 1.0 if wt_seq[i] in "GP" else 0.0
            mut_f = 1.0 if mut_seq[i] in "GP" else 0.0
            changes.append(abs(wt_f - mut_f))
        score = float(np.mean(changes)) if changes else 0.0
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"fallback": True, "flexibility_proxy": True}
        )


class SecondaryStructureIntegrity(Constraint):
    """
    Preservation of secondary structure state upon mutation.

    Source: Kabsch W., Sander C. (1983) Biopolymers 22:2577-2637 (DSSP)
    & Chou-Fasman propensities as fallback

    What it checks: Whether mutations cause secondary structure transitions
                    (e.g., helix -> sheet or coil).

    Evaluation: If DSSP provided: fraction of mutations causing 3-state changes
                Fallback: Chou-Fasman propensity class changes
    """

    def __init__(self):
        super().__init__("ss_integrity", weight=0.5)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        dssp_wt = kwargs.get("dssp_wt")
        dssp_mut = kwargs.get("dssp_mut")

        if dssp_wt is not None:
            transitions = 0
            for i in diffs:
                if i < len(dssp_wt):
                    wt_state = DSSP_3STATE.get(dssp_wt[i], "C")
                    mut_state = DSSP_3STATE.get(
                        dssp_mut[i] if dssp_mut and i < len(dssp_mut) else dssp_wt[i], "C"
                    )
                    if wt_state != mut_state:
                        transitions += 1
            score = transitions / max(len(diffs), 1)
            return ConstraintResult(
                self.name, score, True, self.weight,
                {"transitions": transitions, "source": "DSSP"}
            )

        # Fallback: Chou-Fasman class changes
        ia = np.array([AA_IDX[wt_seq[i]] for i in diffs], dtype=np.int32)
        ib = np.array([AA_IDX[mut_seq[i]] for i in diffs], dtype=np.int32)
        transitions = sum(
            1 for a, b in zip(ia, ib)
            if np.argmax(CHOU_FASMAN[a]) != np.argmax(CHOU_FASMAN[b])
        )
        score = transitions / max(len(diffs), 1)
        return ConstraintResult(
            self.name, score, True, self.weight,
            {"fallback": True, "propensity_transitions": transitions}
        )


def build_level3_panel() -> ProtoPanel:
    """Build Level 3: Structural Integrity panel."""
    return ProtoPanel(
        constraints=[
            FoldIntegrity(),
            AllostericNetworkConservation(),
            ConformationalDynamics(),
            SecondaryStructureIntegrity(),
        ],
        mode=CombinationMode.HYBRID,
        threshold=0.5,
        level_name="Level 3: Structural Integrity",
    )


# ==============================================================================
# SECTION 7: LEVEL 4 — EVOLUTIONARY CONSERVATION
# ==============================================================================
# These constraints require evolutionary data (MSA, co-evolution scores,
# conservation profiles). They fall back to BLOSUM-based proxies.
# ==============================================================================

class EvolutionaryCoupling(Constraint):
    """
    Preservation of evolutionarily coupled residue pairs.

    Source: GREMLIN / EVcouplings (Marks D.S. et al. 2011 PLoS ONE 6:e28766)
    & general DCA literature

    What it checks: Whether mutations hit positions that are evolutionarily
                    coupled (co-varying) with other mutated positions.

    Evaluation: If EC scores provided: max penalty for coupled mutations
                Fallback: contact density x BLOSUM penalty
    """

    def __init__(self):
        super().__init__("evolutionary_coupling", weight=1.0)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        ec = kwargs.get("ec_scores")
        if ec is not None:
            max_pen = 0.0
            for pos in diffs:
                for coupled_pos, strength in ec.get(pos, []):
                    if coupled_pos in diffs:
                        max_pen = max(max_pen, strength * 0.1)
            if max_pen > 0:
                return ConstraintResult(
                    self.name, min(max_pen, 1.0), True, self.weight,
                    {"source": "GREMLIN", "max_coupling_penalty": round(max_pen, 3)}
                )

        # Fallback: BLOSUM + contact proxy
        contact = kwargs.get("contact_map")
        conservation = kwargs.get("conservation_scores")
        penalties = []
        for pos in diffs:
            a, b = wt_seq[pos], mut_seq[pos]
            pen = max(0.0, -BLOSUM62[AA_IDX[a], AA_IDX[b]] / 4.0)
            if conservation and pos in conservation:
                pen *= (1.0 + conservation[pos])
            if contact is not None and pos < contact.shape[0] and np.sum(contact[pos] > 0.5) > 5:
                pen *= 1.2
            penalties.append(pen)
        score = float(np.mean(penalties)) if penalties else 0.0
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"fallback": True, "mean_penalty": round(score, 3)}
        )


class PhylogeneticConservation(Constraint):
    """
    Preservation of phylogenetically conserved positions.

    Source: ConSurf / Rate4Site (Pupko T. et al. 2002 Bioinformatics 18:S71)

    What it checks: Whether mutations occur at positions that are conserved
                    across orthologs (low evolutionary rate).

    Evaluation: If conservation scores provided: mean conservation at mutated sites
                Fallback: BLOSUM-based conservation proxy
    """

    def __init__(self):
        super().__init__("phylogenetic_conservation", weight=0.8)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        cons = kwargs.get("conservation_scores")
        if cons is not None:
            pens = [cons.get(i, 0.3) for i in diffs]
            score = min(float(np.mean(pens)), 1.0)
            return ConstraintResult(
                self.name, score, True, self.weight,
                {"source": "conservation_scores", "mean_conservation": round(float(np.mean(pens)), 3)}
            )

        # Fallback: BLOSUM as conservation proxy
        penalties = []
        for pos in diffs:
            a, b = wt_seq[pos], mut_seq[pos]
            blosum = BLOSUM62[AA_IDX[a], AA_IDX[b]]
            pen = max(0.0, (-blosum + 4.0) / 15.0)
            penalties.append(pen)
        score = float(np.mean(penalties)) if penalties else 0.0
        return ConstraintResult(
            self.name, min(score, 1.0), True, self.weight,
            {"fallback": True, "mean_penalty": round(score, 3)}
        )


def build_level4_panel() -> ProtoPanel:
    """Build Level 4: Evolutionary Conservation panel."""
    return ProtoPanel(
        constraints=[EvolutionaryCoupling(), PhylogeneticConservation()],
        mode=CombinationMode.HYBRID,
        threshold=0.5,
        level_name="Level 4: Evolutionary Conservation",
    )


# ==============================================================================
# SECTION 8: LEVEL 5 — FUNCTIONAL / PHENOTYPIC
# ==============================================================================
# These constraints require functional annotations (catalytic sites,
# binding interfaces, etc.). They use hard gates — any hit is severe.
# ==============================================================================

class CatalyticSiteIntegrity(Constraint):
    """
    Preservation of catalytic/active site residues.

    Source: Catalytic Site Atlas (CSA) / UniProt functional annotations

    What it checks: Whether mutations hit known catalytic residues.
                    Common catalytic residues: H, D, E, C, K, S, Y

    Evaluation: Hard gate. Any hit at annotated catalytic site = severe penalty.
                Fallback: penalty for mutating common catalytic residue types.
    """

    def __init__(self):
        super().__init__("catalytic_integrity", weight=1.0)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        catalytic_sites = kwargs.get("catalytic_sites", [])
        if catalytic_sites:
            hits = sum(1 for i in diffs if i in catalytic_sites)
            if hits > 0:
                return ConstraintResult(
                    self.name, min(hits / len(diffs), 1.0), True, self.weight,
                    {"catalytic_hits": hits, "source": "annotated_sites"}
                )

        # Fallback: common catalytic residue types
        hits = sum(1 for i in diffs if wt_seq[i] in CATALYTIC)
        score = min(hits / max(len(diffs), 1), 1.0) * 0.5
        return ConstraintResult(
            self.name, score, True, self.weight,
            {"fallback": True, "catalytic_type_hits": hits}
        )


class BindingInterfaceDisruption(Constraint):
    """
    Preservation of protein-protein binding interfaces.

    Source: General protein interface analysis literature

    What it checks: Whether mutations disrupt known binding interface residues.

    Evaluation: If interface residues annotated: fraction of mutations at interface
                Fallback: hydrophobicity loss penalty (hydrophobic -> polar at surface)
    """

    def __init__(self):
        super().__init__("binding_interface", weight=0.9)

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> ConstraintResult:
        diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
        if not diffs:
            return ConstraintResult(self.name, 0.0, True, self.weight)

        interface_residues = kwargs.get("interface_residues", [])
        if interface_residues:
            hits = sum(1 for i in diffs if i in interface_residues)
            score = min(hits / max(len(diffs), 1), 1.0)
            return ConstraintResult(
                self.name, score, True, self.weight,
                {"interface_hits": hits, "source": "annotated"}
            )

        # Fallback: hydrophobic-to-polar at surface is bad for interfaces
        score = 0.0
        for i in diffs:
            if wt_seq[i] in HYDROPHOBIC and mut_seq[i] not in HYDROPHOBIC:
                score += 0.15
        score = min(score, 1.0)
        return ConstraintResult(
            self.name, score, True, self.weight,
            {"fallback": True}
        )


def build_level5_panel() -> ProtoPanel:
    """Build Level 5: Functional/Phenotypic panel."""
    return ProtoPanel(
        constraints=[CatalyticSiteIntegrity(), BindingInterfaceDisruption()],
        mode=CombinationMode.HARD_GATES,  # Any hit is severe
        threshold=0.3,
        level_name="Level 5: Functional / Phenotypic",
    )


# ==============================================================================
# SECTION 9: MULTI-LEVEL ORCHESTRATOR
# ==============================================================================

class MultiLevelOrchestrator:
    """
    Orchestrates evaluation across all levels with early stopping,
    level weighting, timing, and certificate generation.

    The certificate system is the key tagging mechanism:
    - Every evaluated protein gets a LevelCertificate
    - cert.tag gives a short string like "L3" or "L1-L3"
    - cert.full_tag gives detailed score breakdown
    - cert.has_passed(level) checks specific level clearance
    """

    def __init__(
        self,
        level_panels: Optional[Dict[int, ProtoPanel]] = None,
        early_stop: bool = True,
        accumulate_scores: bool = False,
        level_weights: Optional[Dict[int, float]] = None,
    ):
        self.early_stop = early_stop
        self.accumulate_scores = accumulate_scores

        if level_panels is None:
            self.level_panels = {
                1: build_level1_panel(),
                2: build_level2_panel(),
                3: build_level3_panel(),
                4: build_level4_panel(),
                5: build_level5_panel(),
            }
        else:
            self.level_panels = dict(level_panels)

        self.level_weights = level_weights or {
            1: 0.0, 2: 0.2, 3: 0.3, 4: 0.3, 5: 0.2
        }

    def evaluate(self, wt_seq: str, mut_seq: str, **kwargs: Any) -> MultiLevelReport:
        """
        Evaluate a WT->mutation pair across all configured levels.

        Returns a MultiLevelReport with a LevelCertificate embedded.
        The certificate.tag tells you exactly which levels passed.
        """
        if len(wt_seq) != len(mut_seq):
            raise ValueError(
                f"Sequence length mismatch: WT={len(wt_seq)} vs MUT={len(mut_seq)}"
            )

        n_mutations = sum(1 for a, b in zip(wt_seq, mut_seq) if a != b)
        reports: List[LevelReport] = []
        passed_levels: List[int] = []
        level_scores: Dict[int, float] = {}

        stopped_level = max(self.level_panels.keys())
        final_score = 0.0
        all_passed = True

        for level in sorted(self.level_panels.keys()):
            panel = self.level_panels[level]
            t0 = time.perf_counter()
            result = panel.evaluate(wt_seq, mut_seq, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000

            passed = result.passed
            report = LevelReport(level, result, passed, elapsed)
            reports.append(report)

            weight = self.level_weights.get(level, 1.0)

            if self.accumulate_scores:
                final_score += result.aggregate_score * weight
            else:
                final_score = max(final_score, result.aggregate_score * weight)

            if passed:
                passed_levels.append(level)
                level_scores[level] = result.aggregate_score

            if not passed:
                all_passed = False
                if self.early_stop:
                    stopped_level = level
                    break

        final_score = min(final_score, 1.0)

        certificate = LevelCertificate(
            passed_levels=tuple(passed_levels),
            stopped_at_level=stopped_level,
            final_score=final_score,
            final_passed=all_passed,
            level_scores=level_scores,
        )

        return MultiLevelReport(
            wt_seq=wt_seq,
            mut_seq=mut_seq,
            n_mutations=n_mutations,
            level_reports=tuple(reports),
            final_score=final_score,
            final_passed=all_passed,
            stopped_at_level=stopped_level,
            certificate=certificate,
            metadata={"early_stopped": not all_passed and self.early_stop},
        )

    def clear_all_caches(self) -> None:
        for panel in self.level_panels.values():
            panel.clear_caches()


# ==============================================================================
# SECTION 10: ADVERSARIAL MUTATION GENERATOR
# ==============================================================================

class MutationStrategy(Enum):
    """Strategies for selecting substitutions."""
    RANDOM = "random"
    BLOSUM_GUIDED = "blosum_guided"
    CONSERVATIVE = "conservative"
    RADICAL = "radical"
    SURFACE_ONLY = "surface_only"


class AdversarialMutator:
    """
    Generates conservative mutations and scores them through the orchestrator.

    Each generated mutant carries a LevelCertificate showing exactly
    which levels it passed.
    """

    def __init__(
        self,
        orchestrator: MultiLevelOrchestrator,
        strategy: MutationStrategy = MutationStrategy.BLOSUM_GUIDED,
        n_positions: int = 1,
        max_attempts: int = 100,
    ):
        self.orchestrator = orchestrator
        self.strategy = strategy
        self.n_positions = n_positions
        self.max_attempts = max_attempts

    def _select_positions(self, seq: str, **kwargs: Any) -> List[int]:
        import random
        n = len(seq)

        if self.strategy == MutationStrategy.SURFACE_ONLY:
            sasa = kwargs.get("sasa_wt")
            if sasa is not None and len(sasa) == n:
                rel_sasa = [sasa[i] / MAX_ASA[AA_IDX[seq[i]]] for i in range(n)]
                candidates = [i for i, r in enumerate(rel_sasa) if r > 0.25]
                if candidates:
                    k = min(self.n_positions, len(candidates))
                    return random.sample(candidates, k)

        return random.sample(range(n), min(self.n_positions, n))

    def _select_substitution(self, wt_aa: str, strategy: MutationStrategy) -> str:
        import random

        if strategy == MutationStrategy.CONSERVATIVE:
            scores = BLOSUM62[AA_IDX[wt_aa]]
            probs = np.maximum(scores, 0) + 0.1
            probs[AA_IDX[wt_aa]] = 0
            probs /= probs.sum()
            return str(np.random.choice(AA_LIST, p=probs))

        elif strategy == MutationStrategy.RADICAL:
            scores = BLOSUM62[AA_IDX[wt_aa]]
            probs = np.maximum(-scores, 0) + 0.1
            probs[AA_IDX[wt_aa]] = 0
            probs /= probs.sum()
            return str(np.random.choice(AA_LIST, p=probs))

        elif strategy == MutationStrategy.BLOSUM_GUIDED:
            scores = BLOSUM62[AA_IDX[wt_aa]].copy()
            scores[AA_IDX[wt_aa]] = -10
            probs = np.exp(scores / 2.0)
            probs /= probs.sum()
            return str(np.random.choice(AA_LIST, p=probs))

        else:
            candidates = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            return random.choice(candidates)

    def generate_mutant(self, wt_seq: str, **kwargs: Any) -> str:
        """Generate a single mutant sequence."""
        positions = self._select_positions(wt_seq, **kwargs)
        mut_seq = list(wt_seq)
        for pos in positions:
            mut_seq[pos] = self._select_substitution(wt_seq[pos], self.strategy)
        return "".join(mut_seq)

    def propose_and_score(
        self, wt_seq: str, n_candidates: int = 10, **kwargs: Any
    ) -> List[Tuple[str, MultiLevelReport]]:
        """
        Generate and score multiple candidate mutants.

        Returns list of (mutant_sequence, report) tuples sorted by final_score.
        Each report contains a certificate with the level tag.
        """
        candidates: List[Tuple[str, MultiLevelReport]] = []
        seen = {wt_seq}

        for _ in range(self.max_attempts):
            if len(candidates) >= n_candidates:
                break

            mut_seq = self.generate_mutant(wt_seq, **kwargs)
            if mut_seq in seen:
                continue
            seen.add(mut_seq)

            report = self.orchestrator.evaluate(wt_seq, mut_seq, **kwargs)
            candidates.append((mut_seq, report))

        candidates.sort(key=lambda x: x[1].final_score)
        return candidates


# ==============================================================================
# SECTION 11: HIGH-LEVEL API
# ==============================================================================

def score_mutation(
    wt_seq: str,
    mut_seq: str,
    levels: Tuple[int, ...] = (1, 2, 3, 4),
    **kwargs: Any,
) -> MultiLevelReport:
    """
    Score a specific mutation through selected levels.

    Returns a MultiLevelReport with embedded LevelCertificate.
    Access the certificate via report.certificate
    """
    panels: Dict[int, ProtoPanel] = {}
    for lvl in levels:
        if lvl == 1:
            panels[lvl] = build_level1_panel()
        elif lvl == 2:
            panels[lvl] = build_level2_panel(
                use_esm2=kwargs.pop("use_esm2", False)
            )
        elif lvl == 3:
            panels[lvl] = build_level3_panel()
        elif lvl == 4:
            panels[lvl] = build_level4_panel()
        elif lvl == 5:
            panels[lvl] = build_level5_panel()

    orchestrator = MultiLevelOrchestrator(
        level_panels=panels,
        early_stop=kwargs.pop("early_stop", True),
    )
    return orchestrator.evaluate(wt_seq, mut_seq, **kwargs)


def suggest_conservative_mutations(
    wt_seq: str,
    n_suggestions: int = 5,
    strategy: MutationStrategy = MutationStrategy.CONSERVATIVE,
    **kwargs: Any,
) -> List[Tuple[str, float, str]]:
    """
    Suggest conservative mutations for a wild-type sequence.

    Returns list of (mutant_sequence, final_score, level_tag) tuples.
    The level_tag (e.g., "L3", "L1-L2", "FAIL-L2") tells you exactly
    which levels each suggestion passed.
    """
    orchestrator = MultiLevelOrchestrator(early_stop=False)
    mutator = AdversarialMutator(
        orchestrator=orchestrator,
        strategy=strategy,
        n_positions=kwargs.pop("n_positions", 1),
    )
    candidates = mutator.propose_and_score(
        wt_seq, n_candidates=n_suggestions * 3, **kwargs
    )

    return [
        (mut, rep.final_score, rep.certificate.tag)
        for mut, rep in candidates[:n_suggestions]
    ]


# ==============================================================================
# SECTION 12: BATCH PROCESSING
# ==============================================================================

def parse_fasta(filepath: str) -> Iterator[Tuple[str, str]]:
    """Parse a FASTA file, yielding (header, sequence) tuples."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")

    current_header = ""
    current_seq: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header and current_seq:
                    yield (current_header, "".join(current_seq))
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

    if current_header and current_seq:
        yield (current_header, "".join(current_seq))


def batch_process(
    input_fasta: str,
    output_json: str,
    levels: Tuple[int, ...] = (1, 2, 3, 4, 5),
    strategy: MutationStrategy = MutationStrategy.CONSERVATIVE,
    n_mutations_per_protein: int = 3,
    n_candidates: int = 10,
    **kwargs: Any,
) -> None:
    """
    Process an entire FASTA database: for each protein, generate conservative
    mutations and score them through all levels. Output is JSON with
    LevelCertificate tags for every mutant.
    """
    orchestrator = MultiLevelOrchestrator(
        early_stop=kwargs.pop("early_stop", False)
    )

    results: List[Dict[str, Any]] = []

    for header, seq in parse_fasta(input_fasta):
        mutator = AdversarialMutator(
            orchestrator=orchestrator,
            strategy=strategy,
            n_positions=n_mutations_per_protein,
        )

        candidates = mutator.propose_and_score(seq, n_candidates=n_candidates, **kwargs)

        protein_result = {
            "protein_id": header,
            "wt_sequence": seq,
            "wt_length": len(seq),
            "mutants": [
                {
                    "mutant_sequence": mut,
                    "n_mutations": rep.n_mutations,
                    "level_certificate": rep.certificate.to_dict(),
                    "final_score": round(rep.final_score, 4),
                    "final_passed": rep.final_passed,
                    "stopped_at_level": rep.stopped_at_level,
                }
                for mut, rep in candidates
            ],
        }
        results.append(protein_result)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} proteins. Results written to {output_json}")


# ==============================================================================
# SECTION 13: CLI ENTRY POINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Level Adversarial Protein Editing Framework v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a specific mutation
  python protein_editor.py --wt MKTLLIL --mut MKTLLIL --levels 1 2 3

  # Suggest conservative mutations
  python protein_editor.py --wt MKTLLIL --suggest 5 --strategy conservative

  # Batch process a FASTA database
  python protein_editor.py --input proteins.fasta --output results.json --levels 1 2 3 4 5

Level Tags:
  L1, L2, L3, L4, L5     = passed exactly that level
  L1-L3                  = passed levels 1 through 3
  FAIL-L2                = failed at level 2
        """,
    )

    parser.add_argument("--wt", help="Wild-type sequence")
    parser.add_argument("--mut", help="Mutant sequence (if scoring a specific mutation)")
    parser.add_argument("--suggest", type=int, default=0, help="Suggest N conservative mutations")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2],
        help="Levels to evaluate (1-5)"
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in MutationStrategy],
        default="blosum_guided",
        help="Mutation selection strategy",
    )
    parser.add_argument(
        "--no-early-stop", action="store_true",
        help="Run all levels even if one fails"
    )
    parser.add_argument(
        "--esm2", action="store_true",
        help="Use ESM2 if available"
    )
    parser.add_argument(
        "--input", help="Input FASTA file for batch processing"
    )
    parser.add_argument(
        "--output", help="Output JSON file for batch processing"
    )
    parser.add_argument(
        "--n-positions", type=int, default=1,
        help="Number of positions to mutate per candidate"
    )

    args = parser.parse_args()

    # Batch mode
    if args.input and args.output:
        batch_process(
            args.input,
            args.output,
            levels=tuple(args.levels),
            strategy=MutationStrategy(args.strategy),
            n_mutations_per_protein=args.n_positions,
            early_stop=not args.no_early_stop,
        )
        return

    # Single mutation scoring
    if args.mut and args.wt:
        report = score_mutation(
            args.wt,
            args.mut,
            levels=tuple(args.levels),
            use_esm2=args.esm2,
            early_stop=not args.no_early_stop,
        )

        print(f"\n{'='*70}")
        print(f"MUTATION REPORT")
        print(f"{'='*70}")
        print(f"WT:       {args.wt}")
        print(f"MUT:      {args.mut}")
        print(f"N mutations: {report.n_mutations}")
        print(f"Final score: {report.final_score:.4f}")
        print(f"Passed:   {report.final_passed}")
        print(f"Certificate: {report.certificate.tag}")
        print(f"Full tag: {report.certificate.full_tag}")

        if not report.final_passed and report.stopped_at_level < max(args.levels):
            print(f"(Early-stopped at Level {report.stopped_at_level})")

        for lr in report.level_reports:
            status = "PASS" if lr.passed else "FAIL"
            print(f"  Level {lr.level}: [{status}] score={lr.panel_result.aggregate_score:.4f} "
                  f"({lr.time_ms:.1f}ms)")
        return

    # Mutation suggestion mode
    if args.suggest > 0 and args.wt:
        suggestions = suggest_conservative_mutations(
            args.wt,
            n_suggestions=args.suggest,
            strategy=MutationStrategy(args.strategy),
            use_esm2=args.esm2,
            n_positions=args.n_positions,
        )

        print(f"\n{'='*70}")
        print(f"TOP {args.suggest} CONSERVATIVE MUTATION SUGGESTIONS")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Score':<8} {'Tag':<10} {'Mutant'}")
        print("-" * 70)
        for i, (mut, score, tag) in enumerate(suggestions, 1):
            print(f"{i:<4} {score:<8.4f} {tag:<10} {mut}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
