"""Central config: where the Paper-1 artifacts live, and where results go.

The Hierarchical-Model-SNN-main artifacts are NOT bundled in this repository.
Point to them by setting the SNN_ARTIFACTS environment variable (the notebook
does this for you), or drop the Hierarchical-Model-SNN-main folder next to this
repository so the default relative path resolves.
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("SNN_ARTIFACTS") or os.path.join(HERE, "..", "Hierarchical-Model-SNN-main", "artifacts")
ART = os.path.abspath(ART) + os.sep                     # trailing separator
RESULTS = os.path.join(HERE, "results")
for _d in (RESULTS, os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables"),
           os.path.join(RESULTS, "_cache")):
    os.makedirs(_d, exist_ok=True)
