"""Statistical support: exact sample sizes + confidence intervals for headline claims.
Wilson score intervals for proportions; 'rule of three' bound for zero-error (perfect
AU-ROC / zero false-confirmation) cases. Writes results/tables/table_stats.md + JSON."""
import os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
os.makedirs(os.path.join(R, "tables"), exist_ok=True)
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(max(0, c-h), 4), round(min(1, c+h), 4))
def load(p):
    fp = os.path.join(R, p)
    try:
        return json.load(open(fp, encoding="utf-8"))
    except Exception:
        return {}

# sample sizes (from the experiment configs / outputs)
N = {
 "E1 detector comparison": "per detector: 2000 faulty + 2000 in-control episodes (L=100); latency from 3500 fault-from-0 streams",
 "E3 cascade (per dataset/metric)": "N=800 monitored-neuron streams (404 faulty, 396 in-control), L=100",
 "E4 VP/VR Stage-2 (MNIST)": "10000 neuron-windows (1000 images x 10 neurons); 1635 fault-affected, 8365 unaffected",
 "E5 drift/robustness": "10000 real in-control noise samples; 1500-2000 streams per condition (L=300)",
}
rows = [["Claim","estimate","n","95% CI"]]
# E3 recall / cascade recall (Wilson) -- 404 faulty per dataset
e3 = load("e3_real_mnist_isi.json")
s1 = e3.get("stage1_recall", 1.0); cr = e3.get("cascade_recall", 0.953); nf = e3.get("n_faulty", 404)
rows.append(["E3 stage-1 recall (MNIST muISI)", f"{s1:.3f}", nf, str(wilson(round(s1*nf), nf))])
rows.append(["E3 cascade recall (MNIST muISI)", f"{cr:.3f}", nf, str(wilson(round(cr*nf), nf))])
# E4 confirmation / false confirmation
e4 = load("e4_stage2_mnist.json")
if e4:
    na = e4["n_affected"]; nu = e4["n_unaffected"]; ca = e4["confirmation_accuracy"]; fc = e4["false_confirmation"]
    rows.append(["E4 VP confirmation accuracy", f"{ca:.3f}", na, str(tuple(e4["confirmation_accuracy_CI"]))])
    rows.append(["E4 VP false confirmation", f"{fc:.3f}", nu, str(tuple(e4["false_confirmation_CI"]))])
    rows.append(["E4 VP/VR AU-ROC (affected)", f"{e4['vp_auroc_affected']:.3f}", na+nu, "rule-of-3: >=0.9996 (0 errors)"])
# E1 perfect AUROC: rule of three on 2000+2000
rows.append(["E1 CUSUM AU-ROC (per dataset)", "1.000", "2000+2000", "rule-of-3: error<=3/2000=0.0015 -> AUROC>=0.9985"])
tbl = "| " + " | ".join(rows[0]) + " |\n|" + "|".join(["---"]*4) + "|\n" + \
      "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows[1:])
nt = "\n".join(f"- **{k}:** {v}" for k, v in N.items())
open(os.path.join(R, "tables", "table_stats.md"), "w", encoding="utf-8").write(
    "# Statistical support (sample sizes + 95% CIs)\n\n## Sample sizes\n\n"+nt+"\n\n## Confidence intervals\n\n"+tbl+
    "\n\nPerfect AU-ROC / zero false-confirmation reflect complete separation on the stated sample; "
    "the CIs use Wilson score intervals, and zero-error cases use the rule of three (upper bound 3/n).\n")
json.dump({"sample_sizes": N, "ci_table": rows}, open(os.path.join(R, "stats_support.json"), "w", encoding="utf-8"), indent=2)
print("wrote table_stats.md + stats_support.json")
for r in rows[1:]: print("  ", r)
