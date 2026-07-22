"""
metrics_cost.py -- Cost algebra for the cheap-first / expensive-on-demand cascade.

Cheap metrics (muISI, CV) cost O(S) per neuron-window; the sensitive
Victor-Purpura / van Rossum distances cost O(S^2). The cascade runs the cheap
metric + CUSUM on every neuron every window and pays the expensive metric only
on the stage-1-flagged fraction p:

    cost_cascade      = N * c_cheap + p * N * c_expensive
    cost_vp_everywhere = N * c_expensive

With c_cheap = S and c_expensive = S^2 the cascade is cheaper than
VP-everywhere for every flag rate below the crossover

    p* = 1 - 1/S

and the savings factor (VP-everywhere / cascade) is

    savings(p) = S / (1 + p*S).

These closed forms are what the verifier pass checks.
"""

from __future__ import annotations

import numpy as np


def cost_cheap(S: float) -> float:
    """Per-neuron-window cost of a cheap O(S) metric (muISI / CV)."""
    return float(S)


def cost_expensive(S: float) -> float:
    """Per-neuron-window cost of an O(S^2) metric (Victor-Purpura / van Rossum)."""
    return float(S) ** 2


def cascade_cost(N: int, S: float, p: float,
                 c_cheap=None, c_exp=None) -> float:
    """Total cost of the cascade at stage-1 flag rate p."""
    cc = cost_cheap(S) if c_cheap is None else c_cheap
    ce = cost_expensive(S) if c_exp is None else c_exp
    return N * cc + p * N * ce


def vp_everywhere_cost(N: int, S: float, c_exp=None) -> float:
    """Total cost of computing the expensive metric on every neuron-window."""
    ce = cost_expensive(S) if c_exp is None else c_exp
    return N * ce


def cheap_everywhere_cost(N: int, S: float, c_cheap=None) -> float:
    """Total cost of the cheap metric everywhere (Paper-1's deployed floor)."""
    cc = cost_cheap(S) if c_cheap is None else c_cheap
    return N * cc


def crossover_flag_rate(S: float) -> float:
    """Flag rate p* at which cascade cost equals VP-everywhere cost: 1 - 1/S."""
    return 1.0 - 1.0 / S


def savings_factor(S: float, p: float) -> float:
    """VP-everywhere cost / cascade cost = S / (1 + p*S)."""
    return S / (1.0 + p * S)


def savings_vs_cheap_floor(S: float, p: float) -> float:
    """Cascade cost / cheap-everywhere cost = 1 + p*S (overhead over the floor)."""
    return 1.0 + p * S


if __name__ == "__main__":
    N, S = 256, 15
    for p in (0.02, 0.05, crossover_flag_rate(S)):
        cas = cascade_cost(N, S, p)
        vp = vp_everywhere_cost(N, S)
        print(f"p={p:6.3f}  cascade={cas:8.0f}  vp_all={vp:8.0f}  "
              f"savings={vp / cas:6.2f}x  (closed-form {savings_factor(S, p):6.2f}x)")
    print("crossover p* =", round(crossover_flag_rate(S), 4))
