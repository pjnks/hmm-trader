"""
ensemble.py
───────────
Ensemble HMM that trains multiple HMMRegimeModels with different n_states
values and uses majority voting to determine regime classification.

This directly addresses the robustness sensitivity finding that only
n_states=6 works in isolation (5 → Sharpe -3.567, 7 → -0.309).  By
requiring 2-of-3 consensus, the ensemble prevents trading when only one
model sees a BULL regime — a structural check against parameter overfitting.

Public API (mirrors HMMRegimeModel interface)
─────────────────────────────────────────────
  ensemble = EnsembleHMM()
  ensemble.fit(df)               # train one HMM per n_states value
  df       = ensemble.predict(df)   # majority-vote regime_cat + consensus confidence
  ensemble.save(path)
  ensemble = EnsembleHMM.load(path)
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.hmm_model import HMMRegimeModel, FEATURE_COLS

log = logging.getLogger(__name__)


class EnsembleHMM:
    """Multi-model HMM ensemble with majority-vote regime classification."""

    def __init__(
        self,
        n_states_list: "list[int] | None" = None,
        cov_type:      str = config.COV_TYPE,
        feature_cols:  "list[str] | None" = None,
        min_agreement: int = config.ENSEMBLE_MIN_AGREEMENT,
        min_converged: int = config.ENSEMBLE_MIN_CONVERGED,
    ) -> None:
        self.n_states_list = n_states_list or list(config.ENSEMBLE_N_STATES)
        self.cov_type      = cov_type
        self.feature_cols  = feature_cols or FEATURE_COLS
        self.min_agreement = min_agreement
        self.min_converged = min_converged

        self.models: list[HMMRegimeModel] = []
        self.converged: bool = False
        self._converged_idx: list[int] = []

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "EnsembleHMM":
        """Train one HMM per n_states value.  Single-threaded, sequential."""
        self.models = []
        self._converged_idx = []

        for i, n_states in enumerate(self.n_states_list):
            model = HMMRegimeModel(
                n_states     = n_states,
                cov_type     = self.cov_type,
                feature_cols = self.feature_cols,
            )
            model.fit(df)
            self.models.append(model)

            if model.converged:
                self._converged_idx.append(i)
            else:
                log.warning("Ensemble member n_states=%d did not converge", n_states)

        n_conv = len(self._converged_idx)
        self.converged = n_conv >= self.min_converged

        log.info("Ensemble fitted: %d/%d models converged (need %d)",
                 n_conv, len(self.models), self.min_converged)
        return self

    # ── Predict ──────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime for each bar using majority vote across converged models.

        Returns df.copy() with columns:
          state       – 0 (placeholder for interface compatibility)
          regime      – lowercase regime_cat (e.g., "bull", "bear", "chop")
          regime_cat  – "BULL" | "BEAR" | "CHOP"  (majority vote)
          confidence  – average posterior of models agreeing with the majority
        """
        if not self.models:
            raise RuntimeError("Call fit() before predict().")

        active = [self.models[i] for i in self._converged_idx]
        if len(active) < self.min_converged:
            raise RuntimeError(
                f"Only {len(active)} models converged, need {self.min_converged}"
            )

        # Collect per-model predictions
        preds = []
        for model in active:
            pred_df = model.predict(df)
            preds.append((
                pred_df["regime_cat"].values,
                pred_df["confidence"].values,
            ))

        n_bars = len(df)
        voted_cats = []
        voted_confs = []

        for i in range(n_bars):
            cats  = [p[0][i] for p in preds]
            confs = [p[1][i] for p in preds]
            cat, conf = self._vote(cats, confs)
            voted_cats.append(cat)
            voted_confs.append(conf)

        out = df.copy()
        out["regime_cat"] = voted_cats
        out["confidence"] = voted_confs
        out["state"]      = 0
        out["regime"]     = [c.lower() for c in voted_cats]

        return out

    # ── Voting ───────────────────────────────────────────────────────────────
    @staticmethod
    def _vote(
        categories: list[str],
        confidences: list[float],
    ) -> tuple[str, float]:
        """
        Majority vote with consensus confidence.

        If >= min_agreement models agree → winner = majority category,
          confidence = average of agreeing models' posteriors.
        If all disagree → CHOP (conservative: no entry), confidence = average all.
        """
        counts = Counter(categories)
        winner, count = counts.most_common(1)[0]

        if count >= 2:
            agreeing_confs = [c for cat, c in zip(categories, confidences)
                             if cat == winner]
            return winner, float(np.mean(agreeing_confs))
        else:
            # All disagree — conservative fallback
            return "CHOP", float(np.mean(confidences))

    # ── Sojourn / Regime Half-Life ──────────────────────────────────────────
    def get_regime_halflife(self, regime_cat: str) -> float:
        """Average regime half-life across converged ensemble models."""
        half_lives = []
        for model in self.models:
            if model.converged:
                hl = model.get_regime_halflife(regime_cat)
                if hl < 100.0:  # Skip default fallback values
                    half_lives.append(hl)
        if half_lives:
            return float(np.mean(half_lives))
        return 100.0  # No valid data — conservative default

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: "str | Path") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Ensemble saved → %s", path)

    @staticmethod
    def load(path: "str | Path") -> "EnsembleHMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("Ensemble loaded ← %s", path)
        return obj
