"""
hmm_model.py
────────────
N-state Gaussian HMM with automatic regime labelling.

Auto-labelling logic
────────────────────
After fitting, states are sorted by observed mean log-return and assigned
labels in three tiers.  The bucket sizes come from config.BULL_STATES and
config.BEAR_STATES; everything in between is CHOP.

  top    BULL_STATES  → bull_strong, bull_mild, …
  bottom BEAR_STATES  → …, bear_mild, bear_crash   (worst return = bear_crash)
  middle (n − B − B)  → chop_high_vol, chop_neutral, chop_low_vol, chop_extra_N …
                         (sub-ranked by HMM emission covariance = volatility)

This assignment is purely data-driven; it works for any n_states ≥ 4 with no
hardcoded assumptions.

Public API
──────────
  model = HMMRegimeModel()
  model.fit(df)               # train on self.feature_cols
  df    = model.predict(df)   # adds state / regime / confidence / regime_cat
  label = model.state_label(state_id)
  cat   = model.state_category(state_id)   # "BULL" | "BEAR" | "CHOP"
  stats = model.get_state_stats()          # per-state summary DataFrame
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

FEATURE_COLS = ["log_return", "price_range", "volume_change"]

# ─── Label pools (assigned dynamically; never used as a fixed-length template) ─
_BULL_LABELS = ["bull_strong", "bull_mild"]
_BEAR_LABELS = ["bear_mild", "bear_crash"]   # index 0 = less-bad, -1 = worst
_CHOP_LABELS = ["chop_high_vol", "chop_neutral", "chop_low_vol"] + \
               [f"chop_extra_{i}" for i in range(20)]

# Flat set of all known labels — used by _label_to_category
_ALL_LABELS = _BULL_LABELS + _CHOP_LABELS + _BEAR_LABELS


class HMMRegimeModel:
    def __init__(
        self,
        n_states:     int = config.N_STATES,
        n_iter:       int = config.N_ITER,
        random_state: int = config.RANDOM_STATE,
        cov_type:     str = config.COV_TYPE,
        feature_cols: "list[str] | None" = None,
    ) -> None:
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.random_state = random_state
        self.cov_type     = cov_type
        self.feature_cols = feature_cols or FEATURE_COLS   # per-instance override

        self._hmm:    Optional[GaussianHMM]   = None
        self._scaler: Optional[StandardScaler] = None
        self.converged: bool = False   # set after fit(); checked by optimizer

        # Populated after fit()
        self._rank_to_label: dict[int, str]  = {}
        self._state_to_rank: dict[int, int]  = {}
        self._state_to_label: dict[int, str] = {}

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "HMMRegimeModel":
        """Fit HMM on self.feature_cols found in *df*.

        Sprint 3.2 improvements:
          - n_init=3: multiple random restarts, pick best log-likelihood
          - Covariance regularization: add ridge (1e-4) when eigenvalue < 1e-6
          - min_observations_per_state: flag non-converged if any state < 5 obs
        """
        X = df[self.feature_cols].values.astype(np.float64)

        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        log.info("Training %d-state GaussianHMM on %d observations …",
                 self.n_states, len(X_scaled))

        # n_init=3: try 3 random starts, keep the one with best log-likelihood
        n_init = 3
        best_hmm    = None
        best_score  = -np.inf
        best_converged = False

        for init_i in range(n_init):
            seed = self.random_state + init_i
            hmm = GaussianHMM(
                n_components    = self.n_states,
                covariance_type = self.cov_type,
                n_iter          = self.n_iter,
                random_state    = seed,
                verbose         = False,
            )
            try:
                hmm.fit(X_scaled)
                score = hmm.score(X_scaled)
                converged = bool(hmm.monitor_.converged)

                if score > best_score:
                    best_score     = score
                    best_hmm       = hmm
                    best_converged = converged

            except Exception as exc:
                log.warning("HMM init %d/%d failed: %s", init_i + 1, n_init, exc)
                continue

        if best_hmm is None:
            # All inits failed — fall back to single attempt
            log.warning("All %d HMM inits failed; using single fallback", n_init)
            best_hmm = GaussianHMM(
                n_components    = self.n_states,
                covariance_type = self.cov_type,
                n_iter          = self.n_iter,
                random_state    = self.random_state,
                verbose         = False,
            )
            best_hmm.fit(X_scaled)
            best_converged = bool(best_hmm.monitor_.converged)

        self._hmm = best_hmm

        # ── Covariance regularization: ridge when eigenvalue < threshold ───
        self._regularize_covariance(threshold=1e-6, ridge=1e-4)

        self.converged = best_converged
        if not self.converged:
            log.warning("HMM did not converge in %d iterations (best of %d inits) "
                        "– consider increasing N_ITER", self.n_iter, n_init)

        # ── Min-observations-per-state guard ─────────────────────────────
        self._check_min_obs_per_state(X_scaled, min_obs=5)

        self._build_label_map(df, X_scaled)
        log.info("HMM fitted.  State labels: %s", self._state_to_label)
        return self

    def _regularize_covariance(self, threshold: float = 1e-6, ridge: float = 1e-4) -> None:
        """Add ridge regularization to covariance if any eigenvalue < threshold.
        Prevents degenerate models with near-singular covariance matrices."""
        if self._hmm is None:
            return

        n_features = self._hmm.means_.shape[1]
        regularized = False

        if self.cov_type == "full":
            for s in range(self.n_states):
                eigs = np.linalg.eigvalsh(self._hmm.covars_[s])
                if eigs.min() < threshold:
                    self._hmm.covars_[s] += ridge * np.eye(n_features)
                    regularized = True
        elif self.cov_type == "diag":
            for s in range(self.n_states):
                diag = np.atleast_1d(self._hmm.covars_[s])
                if diag.min() < threshold:
                    self._hmm.covars_[s] = np.maximum(diag, ridge)
                    regularized = True

        if regularized:
            log.info("Applied ridge regularization (%.1e) to degenerate covariance", ridge)

    def _check_min_obs_per_state(self, X_scaled: np.ndarray, min_obs: int = 5) -> None:
        """Flag model as non-converged if any state has fewer than min_obs observations.
        This catches degenerate HMMs where a state captures <5 bars."""
        if self._hmm is None or np.isnan(self._hmm.startprob_).any():
            return

        try:
            states = self._hmm.predict(X_scaled)
            for s in range(self.n_states):
                n = int((states == s).sum())
                if n < min_obs:
                    log.warning("State %d has only %d observations (< %d) — "
                                "marking as non-converged", s, n, min_obs)
                    self.converged = False
                    return
        except Exception:
            pass  # predict can fail on degenerate models; handled elsewhere

    # ── Predict ──────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add three columns to *df*:
          state       – raw HMM state integer
          regime      – human-readable label string
          confidence  – posterior probability of the predicted state (0-1)
        """
        if self._hmm is None:
            raise RuntimeError("Call fit() before predict().")

        df = df.copy()
        X  = df[self.feature_cols].values.astype(np.float64)
        X_scaled = self._scaler.transform(X)

        states     = self._hmm.predict(X_scaled)
        posteriors = self._hmm.predict_proba(X_scaled)   # shape (T, K)

        df["state"]      = states
        df["regime"]     = [self._state_to_label[s] for s in states]
        df["confidence"] = posteriors[np.arange(len(states)), states]

        # Convenience category column: BULL / BEAR / CHOP
        df["regime_cat"] = df["regime"].map(self._label_to_category)

        return df

    # ── Public helpers ────────────────────────────────────────────────────────
    def state_label(self, state: int) -> str:
        return self._state_to_label.get(state, "unknown")

    def state_category(self, state: int) -> str:
        label = self.state_label(state)
        return self._label_to_category.get(label, "CHOP")

    @property
    def _label_to_category(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for label in _ALL_LABELS:
            if label.startswith("bull"):
                mapping[label] = "BULL"
            elif label.startswith("bear"):
                mapping[label] = "BEAR"
            else:
                mapping[label] = "CHOP"
        return mapping

    def get_state_stats(self) -> pd.DataFrame:
        """Return a per-state summary table (mean of each feature, category)."""
        if self._hmm is None:
            raise RuntimeError("Call fit() first.")
        orig_means = self._scaler.inverse_transform(self._hmm.means_)
        rows = []
        for s in range(self.n_states):
            row: dict = {
                "state":    s,
                "label":    self._state_to_label.get(s, "?"),
                "category": self._label_to_category.get(
                    self._state_to_label.get(s, ""), "?"
                ),
            }
            for i, col in enumerate(self.feature_cols):
                row[f"mean_{col}"] = round(float(orig_means[s, i]), 6)
            rows.append(row)
        return pd.DataFrame(rows).set_index("state")

    # ── Sojourn / Regime Half-Life ──────────────────────────────────────────
    def get_regime_halflife(self, regime_cat: str) -> float:
        """Return expected regime half-life (in bars) from the transition matrix.

        For a geometric sojourn distribution (standard HMM assumption):
            P(duration >= k) = A[i,i]^k
            half_life = -ln(2) / ln(A[i,i])

        If multiple HMM states map to the same regime category (e.g., bull_strong
        and bull_mild both → BULL), their self-transition probabilities are averaged
        weighted by stationary distribution.

        Returns a large default (100 bars) if the transition matrix is unavailable
        or the regime category has no matching states.
        """
        if self._hmm is None or not hasattr(self._hmm, "transmat_"):
            return 100.0

        A = self._hmm.transmat_
        # Find all raw states that map to this regime category
        matching_states = [
            s for s in range(self.n_states)
            if self.state_category(s) == regime_cat
        ]
        if not matching_states:
            return 100.0

        # Weighted average self-transition by stationary distribution
        pi = self._hmm.startprob_
        total_weight = sum(pi[s] for s in matching_states)
        if total_weight <= 0:
            total_weight = len(matching_states)
            weights = {s: 1.0 / total_weight for s in matching_states}
        else:
            weights = {s: pi[s] / total_weight for s in matching_states}

        avg_self_trans = sum(A[s, s] * weights[s] for s in matching_states)

        if avg_self_trans <= 0 or avg_self_trans >= 1.0:
            return 100.0

        half_life = -np.log(2) / np.log(avg_self_trans)
        return max(1.0, float(half_life))

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Model saved → %s", path)

    @staticmethod
    def load(path: str | Path) -> "HMMRegimeModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("Model loaded ← %s", path)
        return obj

    # ── Internal ─────────────────────────────────────────────────────────────
    def _build_label_map(self, df: pd.DataFrame, X_scaled: np.ndarray) -> None:
        """
        Assign BULL / CHOP / BEAR labels to all states purely by rank.

        Algorithm
        ─────────
        1. Compute each state's mean log-return from observed bars.
           Falls back to HMM emission mean if log_return not in df.
        2. Sort states descending by mean return → `ranked`.
        3. ranked[:n_bull]  → bull labels (best return first)
           ranked[-n_bear:] → bear labels (less-bad first, worst = bear_crash)
           middle states     → chop labels sub-ranked by HMM covariance magnitude
                               (highest covariance = chop_high_vol, etc.)
        4. No hardcoded length assumptions — works for any n_states ≥ 4.
        """
        n_bull = config.BULL_STATES
        n_bear = config.BEAR_STATES
        n_chop = self.n_states - n_bull - n_bear

        # ── 1. Mean return per state ──────────────────────────────────────────
        # Primary path: Viterbi-decode training data, compute observed mean
        # log_return per state.  Falls back to HMM emission means if predict()
        # fails (e.g. degenerate model with NaN startprob_).
        orig_means = self._scaler.inverse_transform(self._hmm.means_)

        raw_states = None
        if not np.isnan(self._hmm.startprob_).any():
            try:
                raw_states = self._hmm.predict(X_scaled)
            except Exception as exc:
                log.warning("predict() failed in _build_label_map: %s — "
                            "falling back to emission means", exc)

        if raw_states is not None and "log_return" in df.columns:
            log_ret_vals = df["log_return"].values
            mean_returns = []
            for s in range(self.n_states):
                mask = raw_states == s
                mean_returns.append(
                    float(log_ret_vals[mask].mean()) if mask.any() else 0.0
                )
        else:
            # Fallback: use HMM emission mean of log_return if it is a feature,
            # otherwise use the first feature as a return proxy.
            ret_idx = (self.feature_cols.index("log_return")
                       if "log_return" in self.feature_cols else 0)
            mean_returns = [float(orig_means[s, ret_idx])
                            for s in range(self.n_states)]
            log.warning("Using HMM emission means for state return ranking "
                        "(observed log_return unavailable or predict failed)")

        # ── 2. Sort states by mean return (descending) ───────────────────────
        ranked = sorted(range(self.n_states),
                        key=lambda s: mean_returns[s], reverse=True)

        bull_states = ranked[:n_bull]
        chop_states = ranked[n_bull: self.n_states - n_bear]
        bear_states = ranked[self.n_states - n_bear:]   # worst return is last

        # ── 3a. Chop sub-ranking by emission covariance (proxy for volatility) ─
        # For "full" covariance covars_ is (K, F, F); for "diag" it is (K, F).
        try:
            if self.cov_type == "full":
                state_spread = {s: float(self._hmm.covars_[s].diagonal().mean())
                                for s in range(self.n_states)}
            else:
                state_spread = {s: float(np.atleast_1d(self._hmm.covars_[s]).mean())
                                for s in range(self.n_states)}
        except Exception:
            # Degenerate model — use absolute emission mean as spread proxy
            state_spread = {s: float(abs(orig_means[s]).mean())
                            for s in range(self.n_states)}

        chop_by_vol = sorted(chop_states,
                             key=lambda s: state_spread[s], reverse=True)

        # ── 3b. Build final label maps ────────────────────────────────────────
        self._rank_to_label  = {}
        self._state_to_rank  = {}
        self._state_to_label = {}

        for rank, state in enumerate(bull_states):
            label = _BULL_LABELS[rank]
            self._rank_to_label[rank]   = label
            self._state_to_rank[state]  = rank
            self._state_to_label[state] = label

        # Chop: preserve return-rank for _state_to_rank; label by volatility rank
        vol_rank_of = {s: i for i, s in enumerate(chop_by_vol)}
        for rank_offset, state in enumerate(chop_states):
            rank  = n_bull + rank_offset
            label = _CHOP_LABELS[vol_rank_of[state]]
            self._rank_to_label[rank]   = label
            self._state_to_rank[state]  = rank
            self._state_to_label[state] = label

        # Bear: bear_states[0] = less-bad (bear_mild), bear_states[-1] = worst (bear_crash)
        for i, state in enumerate(bear_states):
            rank  = n_bull + n_chop + i
            label = _BEAR_LABELS[i]
            self._rank_to_label[rank]   = label
            self._state_to_rank[state]  = rank
            self._state_to_label[state] = label

        log.debug("State ranking (best→worst): %s", ranked)
        log.debug("Mean returns: %s",
                  {s: f"{r:.5f}" for s, r in enumerate(mean_returns)})
        log.debug("Labels: %s", self._state_to_label)
