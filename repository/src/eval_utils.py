"""
eval_utils.py -- ROC / TPR@FPR helpers and episode-level detection evaluation.

A detector is evaluated at the *episode* level: on an episode of length L it
either raises an alarm (within L windows) or not. Sweeping the detector's
threshold traces an ROC whose axes are

    FPR = P(alarm | in-control episode)
    TPR = P(alarm | faulty  episode).

We also report TPR at 1% FPR and the detection latency (mean windows-to-alarm on
faulty episodes) at a threshold calibrated to a target in-control ARL0.
"""
from __future__ import annotations
import numpy as np


def _alarm_and_delay(statistic_cross, L):
    """Given per-episode first-cross index (-1 if none), return (alarmed, delay)."""
    alarmed = statistic_cross >= 0
    delay = np.where(alarmed, statistic_cross, L)
    return alarmed, delay


def roc_from_thresholds(make_detector, thresholds, pos_streams, neg_streams):
    """Trace an ROC by sweeping detector thresholds.

    make_detector(h) -> Detector with .run_batch. Returns fpr, tpr arrays sorted
    by fpr, plus the raw per-threshold (fpr, tpr, mean_delay_pos).
    """
    L = pos_streams.shape[1]
    rows = []
    for h in thresholds:
        d = make_detector(h)
        pos_cross = d.run_batch(pos_streams)
        neg_cross = d.run_batch(neg_streams)
        tpr = (pos_cross >= 0).mean()
        fpr = (neg_cross >= 0).mean()
        pa, pd = _alarm_and_delay(pos_cross, L)
        mean_delay = pd[pa].mean() if pa.any() else np.nan
        rows.append((fpr, tpr, mean_delay, h))
    rows.sort(key=lambda r: r[0])
    fpr = np.array([r[0] for r in rows])
    tpr = np.array([r[1] for r in rows])
    return fpr, tpr, rows


def auroc(fpr, tpr):
    """Area under the ROC, with (0,0) and (1,1) endpoints enforced."""
    f = np.concatenate([[0.0], fpr, [1.0]])
    t = np.concatenate([[0.0], tpr, [1.0]])
    order = np.argsort(f)
    f, t = f[order], t[order]
    return float(np.trapz(t, f))


def tpr_at_fpr(fpr, tpr, target_fpr=0.01):
    """Interpolate TPR at a target FPR."""
    order = np.argsort(fpr)
    f, t = fpr[order], tpr[order]
    return float(np.interp(target_fpr, f, t))


def single_window_auc(delta, sd):
    """Analytic single-sample detection AUC for a mean shift delta vs noise sd.

    Used by the E2 conditioning study: AUC = Phi(delta / (sd*sqrt(2))).
    """
    from math import erf, sqrt
    z = delta / (sd * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))
