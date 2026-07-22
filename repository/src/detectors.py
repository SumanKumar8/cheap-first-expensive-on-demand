"""
detectors.py -- Stage-1 change detectors for concurrent testing of spiking
neuromorphic hardware.

Implements the four stage-1 detectors compared in the paper, all sharing a
single ARL0-matching calibration harness so that latency is compared at an
*equal false-alarm rate*:

  * PerWindowThreshold  -- Paper-1's Shewhart-type residual test (fires when a
                           single window's standardized residual exceeds h).
  * CUSUM               -- sequential cumulative-sum test, textbook-optimal for
                           small persistent mean shifts.
  * EWMA                -- exponentially-weighted moving-average control chart.
  * ResidualCUSUM       -- subtract a cheap conditional estimate of the metric
                           (linear fit or binned lookup) then run CUSUM on the
                           residual; the "middle ground" between a raw control
                           chart and the learned model.

All detectors operate on a *standardized* residual stream x_t (in-control
x_t ~ N(0,1)); a fault adds a persistent mean shift delta. Everything is
vectorized over M independent streams for fast Monte-Carlo ARL estimation.

Author: cheap-first-escalation project (codename: cheapfirst)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Detector base + implementations
# ---------------------------------------------------------------------------
class Detector:
    """Base class. Subclasses implement run_batch on standardized streams."""

    name: str = "detector"
    threshold: float = 0.0

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        """Run the detector on M streams of length L.

        Parameters
        ----------
        X : (M, L) float array
            Standardized residual streams (in-control ~ N(0, 1)).

        Returns
        -------
        alarm_time : (M,) int array
            Index of the first window that raises an alarm, or -1 if the
            detector never alarms within the stream.
        """
        raise NotImplementedError

    # convenience -------------------------------------------------------
    def run_lengths(self, X: np.ndarray) -> np.ndarray:
        """Run length (windows until alarm), censored at L if no alarm."""
        L = X.shape[1]
        at = self.run_batch(X)
        return np.where(at < 0, L, at + 1)


class PerWindowThreshold(Detector):
    """Shewhart / per-window test -- Paper-1's residual comparison |gamma-ghat|>th.

    One-sided (upper) by default, matched to a positive persistent shift; set
    two_sided=True to fire on |x_t| > h (Paper-1's absolute-residual form).
    """

    def __init__(self, h: float, two_sided: bool = False):
        self.threshold = float(h)
        self.two_sided = two_sided
        self.name = "per-window threshold"

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        if self.two_sided:
            hit = np.abs(X) > self.threshold
        else:
            hit = X > self.threshold
        return _first_hit(hit)


class CUSUM(Detector):
    """One-sided upper CUSUM: S_t = max(0, S_{t-1} + x_t - k); alarm if S_t > h.

    Reference value k = delta_target / 2 is the SPRT-optimal slack for a target
    shift delta_target. two_sided=True runs symmetric upper/lower arms.
    """

    def __init__(self, k: float, h: float, two_sided: bool = False):
        self.k = float(k)
        self.threshold = float(h)
        self.two_sided = two_sided
        self.name = "CUSUM"

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        M, L = X.shape
        S = np.zeros(M)
        Sn = np.zeros(M) if self.two_sided else None
        alarm = np.full(M, -1)
        live = np.ones(M, dtype=bool)
        k, h = self.k, self.threshold
        for t in range(L):
            xt = X[:, t]
            S = np.maximum(0.0, S + xt - k)
            fire = S > h
            if self.two_sided:
                Sn = np.maximum(0.0, Sn - xt - k)
                fire = fire | (Sn > h)
            newly = live & fire
            alarm[newly] = t
            live &= ~fire
            if not live.any():
                break
        return alarm


class EWMA(Detector):
    """EWMA control chart: z_t = lam*x_t + (1-lam)*z_{t-1}; alarm if z_t > L*sigma_z.

    threshold stores the control-limit multiplier L; the steady-state sigma is
    sqrt(lam / (2 - lam)). One-sided upper by default.
    """

    def __init__(self, lam: float, L: float, two_sided: bool = False):
        self.lam = float(lam)
        self.threshold = float(L)  # control-limit multiplier
        self.two_sided = two_sided
        self.name = "EWMA"

    @property
    def sigma_ss(self) -> float:
        lam = self.lam
        return np.sqrt(lam / (2.0 - lam))

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        M, L = X.shape
        z = np.zeros(M)
        alarm = np.full(M, -1)
        live = np.ones(M, dtype=bool)
        lam = self.lam
        cl = self.threshold * self.sigma_ss
        for t in range(L):
            z = lam * X[:, t] + (1.0 - lam) * z
            fire = z > cl
            if self.two_sided:
                fire = fire | (z < -cl)
            newly = live & fire
            alarm[newly] = t
            live &= ~fire
            if not live.any():
                break
        return alarm


class ResidualCUSUM(Detector):
    """CUSUM run on the residual of a cheap conditional estimate of the metric.

    Given paired (input feature u_t, metric m_t), subtract a conditional
    estimate m_hat(u_t) -- a linear fit or a binned lookup learned on in-control
    data -- standardize, then run CUSUM. On an already-standardized/conditioned
    stream this reduces exactly to CUSUM; the conditioning matters only when the
    input drives metric variance (see experiments/e2_conditioning.py).
    """

    def __init__(self, k: float, h: float, conditioner: Optional["Conditioner"] = None):
        self.cusum = CUSUM(k, h, two_sided=False)
        self.conditioner = conditioner
        self.threshold = float(h)
        self.k = float(k)
        self.name = "residual-CUSUM"

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        # X is assumed already standardized/conditioned here.
        return self.cusum.run_batch(X)

    def run_paired(self, U: np.ndarray, M_metric: np.ndarray,
                   mu0: float, sd0: float) -> np.ndarray:
        """Run on raw (input, metric) pairs by conditioning first."""
        if self.conditioner is None:
            resid = M_metric
        else:
            resid = M_metric - self.conditioner.predict(U)
        Z = (resid - mu0) / (sd0 + 1e-12)
        return self.cusum.run_batch(Z)


# ---------------------------------------------------------------------------
# Cheap conditioners (for residual-CUSUM and the E2 conditioning study)
# ---------------------------------------------------------------------------
class Conditioner:
    def fit(self, U: np.ndarray, m: np.ndarray) -> "Conditioner":
        raise NotImplementedError

    def predict(self, U: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearConditioner(Conditioner):
    """m_hat = a*u + b via least squares on in-control data (O(1) params)."""

    def __init__(self):
        self.a = 0.0
        self.b = 0.0

    def fit(self, U, m):
        u = np.asarray(U, float).ravel()
        y = np.asarray(m, float).ravel()
        A = np.vstack([u, np.ones_like(u)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.a, self.b = float(coef[0]), float(coef[1])
        return self

    def predict(self, U):
        u = np.asarray(U, float)
        return self.a * u + self.b


class BinnedConditioner(Conditioner):
    """m_hat = mean(m | bin(u)); a cheap non-parametric lookup table."""

    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins
        self.edges = None
        self.table = None
        self.global_mean = 0.0

    def fit(self, U, m):
        u = np.asarray(U, float).ravel()
        y = np.asarray(m, float).ravel()
        self.global_mean = float(y.mean())
        self.edges = np.quantile(u, np.linspace(0, 1, self.n_bins + 1))
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        idx = np.clip(np.digitize(u, self.edges[1:-1]), 0, self.n_bins - 1)
        table = np.full(self.n_bins, self.global_mean)
        for b in range(self.n_bins):
            sel = idx == b
            if sel.any():
                table[b] = y[sel].mean()
        self.table = table
        return self

    def predict(self, U):
        u = np.asarray(U, float)
        idx = np.clip(np.digitize(u, self.edges[1:-1]), 0, self.n_bins - 1)
        return self.table[idx]


# ---------------------------------------------------------------------------
# ARL0-matching calibration harness
# ---------------------------------------------------------------------------
def _first_hit(hit: np.ndarray) -> np.ndarray:
    """Given a boolean (M, L) hit matrix, return first True column per row (-1 if none)."""
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1)          # 0 if no hit -> mask below
    return np.where(any_hit, first, -1)


def average_run_length(detector: Detector, streams: np.ndarray) -> float:
    """Mean run length (ARL) over a bank of streams, censored at stream length."""
    return float(detector.run_lengths(streams).mean())


def calibrate_arl0(make_detector: Callable[[float], Detector],
                   target_arl0: float,
                   incontrol_streams: np.ndarray,
                   lo: float, hi: float,
                   tol: float = 0.5,
                   max_iter: int = 40) -> Detector:
    """Bisection on a detector's threshold so that in-control ARL == target_arl0.

    make_detector(h) must return a Detector whose threshold is h (or the control
    limit multiplier for EWMA). ARL is monotone increasing in the threshold, so
    bisection converges. Uses common random numbers (a fixed stream bank) for a
    smooth, monotone objective.
    """
    def arl0(h):
        return average_run_length(make_detector(h), incontrol_streams)

    lo_v, hi_v = lo, hi
    # expand hi until ARL0(hi) >= target
    it = 0
    while arl0(hi_v) < target_arl0 and it < 20:
        hi_v *= 1.5
        it += 1
    for _ in range(max_iter):
        mid = 0.5 * (lo_v + hi_v)
        a = arl0(mid)
        if abs(a - target_arl0) <= tol:
            return make_detector(mid)
        if a < target_arl0:
            lo_v = mid
        else:
            hi_v = mid
    return make_detector(0.5 * (lo_v + hi_v))


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------
def incontrol_streams(M: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """M in-control standardized streams ~ N(0, 1)."""
    return rng.standard_normal((M, L))


def shift_streams(M: int, L: int, delta: float, rng: np.random.Generator,
                  change_point: int = 0) -> np.ndarray:
    """M streams with a persistent mean shift delta starting at change_point."""
    X = rng.standard_normal((M, L))
    X[:, change_point:] += delta
    return X


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cal = incontrol_streams(4000, 2000, rng)
    target = 200.0
    delta = 0.75

    shew = calibrate_arl0(lambda h: PerWindowThreshold(h), target, cal, 1.0, 4.0)
    cus = calibrate_arl0(lambda h: CUSUM(delta / 2, h), target, cal, 1.0, 8.0)
    ewm = calibrate_arl0(lambda h: EWMA(0.2, h), target, cal, 1.0, 4.0)

    test = shift_streams(20000, 2000, delta, np.random.default_rng(1))
    for d in (shew, cus, ewm):
        arl0 = average_run_length(d, cal)
        arl1 = average_run_length(d, test)
        print(f"{d.name:22s} thr={d.threshold:6.3f}  ARL0={arl0:6.1f}  ARL1={arl1:6.2f}")
