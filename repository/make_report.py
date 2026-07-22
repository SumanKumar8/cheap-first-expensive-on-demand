"""Aggregate real E1/E2/E3 JSONs across the 3 datasets -> master JSON, tables, figures."""
import os, json, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src"); import metrics_cost as mc

R = "results"; FIG = R+"/figures"; TAB = R+"/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
DS = ["fashionmnist", "mnist", "svhn"]; DSL = {"fashionmnist":"FMNIST","mnist":"MNIST","svhn":"SVHN"}
MET = ["isi", "cv"]; SPW = 15.0
def load(p): return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

# ---- collect ----
M = {"datasets": DS, "e1": {}, "e1_head": {}, "e2": {}, "e3": {}, "golden": {}}
PW = "per-window threshold, raw residual (Paper 1)"; CUR = "CUSUM, raw residual"; CUM = "CUSUM, ML residual"
for ds in DS:
    M["e1"][ds] = {}; M["e1_head"][ds] = {}; M["e3"][ds] = {}
    for m in MET:
        e1 = load(f"{R}/e1_real_{ds}_{m}.json")
        if e1:
            M["e1"][ds][m] = e1
            hd = dict(pw_auroc=e1[PW]["auroc"], pw_lat=e1[PW]["latency_windows"],
                      cusum_auroc=e1[CUR]["auroc"], cusum_lat=e1[CUR]["latency_windows"])
            hd["speedup"] = round(hd["pw_lat"]/hd["cusum_lat"], 2)
            if CUM in e1: hd["cond_gain"] = round(e1[CUM]["auroc"]-e1[CUR]["auroc"], 3)
            M["e1_head"][ds][m] = hd
        e3 = load(f"{R}/e3_real_{ds}_{m}.json")
        if e3: M["e3"][ds][m] = e3
    e2 = load(f"{R}/e2_real_{ds}.json")
    if e2: M["e2"][ds] = e2

# ---- golden MAE fidelity: retrained RF vs Paper-1 shipped margins ----
# epsilon_thrs.npz holds the shipped margins; if absent (cache-only repo)
# fall back to results/golden_shipped.json (same values, cached).
_shipped_cache = load(f"{R}/golden_shipped.json") or {}
for ds in DS:
    s = load(f"{R}/real_pools_{ds}.summary.json")
    folder = {"fashionmnist":"FashionMNIST_Experiments","mnist":"MNIST_Experiments","svhn":"SVHN_Experiments"}[ds]
    try:
        ep = np.load(f"../Hierarchical-Model-SNN-main/artifacts/{folder}/epsilon_thrs.npz", allow_pickle=True)
        ship_by_m = {"isi": [ep["dict3"].item()["isi"][c]["RandomForest"][0] for c in range(10)],
                     "cv":  [ep["dict4"].item()["cv"][c]["RandomForest"][0] for c in range(10)]}
    except FileNotFoundError:
        ship_by_m = {"isi": _shipped_cache[ds]["isi"], "cv": _shipped_cache[ds]["cv"]}
    g = {}
    for m in ("isi", "cv"):
        mine = [s[f"{m}_{c}"]["mae_ml"] for c in range(10)]
        ship = ship_by_m[m]
        g[m] = dict(mine=[round(x,4) for x in mine], shipped=[round(x,4) for x in ship],
                    mean_rel_err=round(float(np.mean([abs(mine[c]-ship[c])/ship[c] for c in range(10)])), 4))
    M["golden"][ds] = g
json.dump(M, open(f"{R}/real_summary_all.json", "w", encoding="utf-8"), indent=1)

# ---- tables ----
def w(path, txt): open(path, "w", encoding="utf-8").write(txt)
# E1 headline
rows = ["| dataset | metric | per-window AUROC | per-window lat | CUSUM AUROC | CUSUM lat | latency speedup | cond. gain (ML-raw) |",
        "|---|---|---|---|---|---|---|---|"]
for ds in DS:
    for m in MET:
        h = M["e1_head"][ds].get(m)
        if h:
            cg = f"{h['cond_gain']:+.3f}" if "cond_gain" in h else "N/A (raw-only)"
            rows.append(f"| {DSL[ds]} | {m} | {h['pw_auroc']:.3f} | {h['pw_lat']:.1f} | {h['cusum_auroc']:.3f} | {h['cusum_lat']:.1f} | {h['speedup']:.2f}x | {cg} |")
w(f"{TAB}/table_e1.md", "# E1 -- Stage-1 detector comparison (real, matched ARL0=200)\n\n"+"\n".join(rows)+"\n")
# E2
r2 = ["| dataset | metric | AUROC raw | AUROC linear | AUROC ML | conditioning gain (ML-raw) |","|---|---|---|---|---|---|"]
for ds in ["fashionmnist","mnist"]:
    if ds in M["e2"]:
        for m in MET:
            e = M["e2"][ds][m]; r2.append(f"| {DSL[ds]} | {m} | {e['auroc_raw']:.3f} | {e['auroc_lin']:.3f} | {e['auroc_ml']:.3f} | {e['gain_ml_minus_raw']:+.3f} |")
w(f"{TAB}/table_e2.md", "# E2 -- Conditioning study (real): does the learned model help?\n\n"+"\n".join(r2)+"\n")
# E3
r3 = ["| dataset | metric | stage-1 recall | cascade recall | latency (win) | false-esc | flag (once) | savings (once) | flag (reset) | savings (reset) |","|---|---|---|---|---|---|---|---|---|---|"]
for ds in DS:
    for m in MET:
        e = M["e3"][ds].get(m)
        if e: r3.append(f"| {DSL[ds]} | {m} | {e['stage1_recall']:.2f} | {e['cascade_recall']:.2f} | {e['mean_latency']} | {e['false_escalation_rate']*100:.2f}% | {e['flag_rate_once']*100:.2f}% | {e['savings_once']:.1f}x | {e['flag_rate']*100:.1f}% | {e['savings_vs_vp']:.1f}x |")
w(f"{TAB}/table_e3.md", "# E3 -- Cheap-first cascade cost (real, model-free stage-1, S=15)\n\n"+"\n".join(r3)+"\n")
# golden
rg = ["| dataset | metric | mean |rel MAE err| vs Paper-1 shipped |","|---|---|---|"]
for ds in DS:
    for m in MET: rg.append(f"| {DSL[ds]} | {m} | {M['golden'][ds][m]['mean_rel_err']*100:.1f}% |")
w(f"{TAB}/table_golden.md", "# Golden-model fidelity: retrained RF vs shipped epsilon_thrs.npz\n\n"+"\n".join(rg)+"\n")

# ---- figures ----
C = {"pw":"#c44","cusum":"#3a7","ml":"#37b","raw":"#888","lin":"#3a7"}
# fig E1: latency + AUROC (muISI) across datasets
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4)); x = np.arange(len(DS)); wd = 0.38
pwl = [M["e1_head"][d]["isi"]["pw_lat"] for d in DS]; cul = [M["e1_head"][d]["isi"]["cusum_lat"] for d in DS]
ax[0].bar(x-wd/2, pwl, wd, label="per-window (Paper 1)", color=C["pw"])
ax[0].bar(x+wd/2, cul, wd, label="CUSUM (cheap-first)", color=C["cusum"])
for i, d in enumerate(DS): ax[0].text(i, max(pwl[i],cul[i])+0.4, f"{M['e1_head'][d]['isi']['speedup']:.1f}x", ha="center", fontsize=9, fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels([DSL[d] for d in DS]); ax[0].set_ylabel("detection latency (windows)")
ax[0].set_title("Stage-1 latency (muISI) @ matched ARL0=200"); ax[0].legend()
pwa = [M["e1_head"][d]["isi"]["pw_auroc"] for d in DS]; cua = [M["e1_head"][d]["isi"]["cusum_auroc"] for d in DS]
ax[1].bar(x-wd/2, pwa, wd, label="per-window (Paper 1)", color=C["pw"])
ax[1].bar(x+wd/2, cua, wd, label="CUSUM (cheap-first)", color=C["cusum"])
ax[1].set_xticks(x); ax[1].set_xticklabels([DSL[d] for d in DS]); ax[1].set_ylim(0.5, 1.02); ax[1].set_ylabel("AU-ROC")
ax[1].set_title("Stage-1 AU-ROC (muISI)"); ax[1].legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_e1_detectors.png", dpi=130); plt.close()
# fig E2: conditioning (muISI) raw/lin/ml for fmnist,mnist
fig, a = plt.subplots(figsize=(6.4, 4.2)); dse = [d for d in DS if d in M["e2"]]; x = np.arange(len(dse))
for k, off, c in [("auroc_raw",-0.27,C["raw"]),("auroc_lin",0,C["lin"]),("auroc_ml",0.27,C["ml"])]:
    a.bar(x+off, [M["e2"][d]["isi"][k] for d in dse], 0.26, label=k.replace("auroc_",""), color=c)
for i,d in enumerate(dse): a.text(i, 1.005, f"gain {M['e2'][d]['isi']['gain_ml_minus_raw']:+.3f}", ha="center", fontsize=8)
a.set_xticks(x); a.set_xticklabels([DSL[d] for d in dse]); a.set_ylim(0.8, 1.03); a.set_ylabel("CUSUM AU-ROC (muISI)")
a.set_title("E2: model-free vs linear vs ML residual"); a.legend(loc="lower left")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_e2_conditioning.png", dpi=130); plt.close()
# fig E3: savings curve + operating points
fig, a = plt.subplots(figsize=(7, 4.6)); p = np.linspace(0.001, 0.32, 200)
a.plot(p, SPW/(1+p*SPW), "k-", lw=1.5, label="savings = S/(1+pS), S=15")
mk = {"fashionmnist":"o","mnist":"s","svhn":"^"}
for d in DS:
    e = M["e3"][d]["isi"]
    a.scatter([e["flag_rate_once"]], [e["savings_once"]], marker=mk[d], s=80, color="#2a7", zorder=3)
    a.scatter([e["flag_rate"]], [e["savings_vs_vp"]], marker=mk[d], s=80, color="#c44", zorder=3)
    a.annotate(DSL[d], (e["flag_rate_once"], e["savings_once"]), textcoords="offset points", xytext=(6,4), fontsize=8)
a.scatter([],[],marker="o",color="#2a7",label="escalate-once policy")
a.scatter([],[],marker="o",color="#c44",label="reset-on-alarm policy")
a.set_xlabel("stage-1 flag rate p"); a.set_ylabel("cost savings vs VP-everywhere"); a.set_ylim(0, 15)
a.set_title("E3: cascade cost savings (muISI, all datasets)"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_e3_cascade.png", dpi=130); plt.close()
# fig golden fidelity scatter
fig, a = plt.subplots(figsize=(5.2, 5)); 
for ds,col in [("fashionmnist","#37b"),("mnist","#e83"),("svhn","#2a7")]:
    for m,mk2 in [("isi","o"),("cv","s")]:
        g=M["golden"][ds][m]; a.scatter(g["shipped"], g["mine"], marker=mk2, alpha=0.7, color=col, label=f"{DSL[ds]} {m}")
lim=[0,5]; a.plot(lim,lim,"k--",lw=0.8); a.set_xlim(0,5); a.set_ylim(0,5)
a.set_xlabel("Paper-1 shipped val MAE"); a.set_ylabel("retrained golden-model val MAE")
a.set_title("Golden-model fidelity (y=x ideal)"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_golden_fidelity.png", dpi=130); plt.close()
print("wrote master JSON, 4 tables, 4 figures")
print("E1 headline speedups (isi):", {DSL[d]: M["e1_head"][d]["isi"]["speedup"] for d in DS})
print("E3 savings once (isi):", {DSL[d]: M["e3"][d]["isi"]["savings_once"] for d in DS})
