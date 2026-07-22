"""Run the Stage-2 Victor-Purpura / van Rossum confirmation (E4) on all three
datasets and emit per-dataset JSON + a combined table.

MNIST uses the released real paired nominal/injected outputs.
FashionMNIST & SVHN use MNIST-DERIVED paired outputs (the MNIST-measured fault
model applied to each dataset's real fault-free output); these live in each
dataset's Nominal/ folder as MNISTderived_{nominal,injected}.txt.
Same E4 code path for all three.
"""
import os, sys, json, time, math
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from config import ART, RESULTS
WIN,NC=110,10
def read_bits(p):
    raw=open(p,'rb').read().replace(b'\r',b''); L=[x for x in raw.split(b'\n') if x]
    from collections import Counter
    W=Counter(len(x) for x in L).most_common(1)[0][0]; L=[x for x in L if len(x)==W]
    return (np.frombuffer(b''.join(L),np.uint8).reshape(len(L),W)-ord('0')).astype(np.uint8)
def parse(p):
    m=read_bits(p)[15:-15]; n=m.shape[0]//WIN; return m[:n*WIN].reshape(n,WIN,NC)
def spk(c): return np.where(c==1)[0].astype(float)+1.0
def avg_isi(t):
    if len(t)==0: return 0.0
    if len(t)==1: return float(t[0])
    return float(np.mean(np.diff(t)))
def van_rossum(a,b,tau=10.0):
    def S(x,y):
        if len(x)==0 or len(y)==0: return 0.0
        return float(np.exp(-np.abs(x[:,None]-y[None,:])/tau).sum())
    return math.sqrt(max(0.0,(S(a,a)+S(b,b)-2*S(a,b))/2.0))
def victor_purpura(a,b,q=0.1):
    n,m=len(a),len(b)
    if n==0 or m==0: return float(max(n,m))
    prev=np.arange(m+1,dtype=float)
    for i in range(1,n+1):
        cur=np.empty(m+1);cur[0]=i;ai=a[i-1]
        for j in range(1,m+1): cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+q*abs(ai-b[j-1]))
        prev=cur
    return float(prev[m])
def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))
def auroc(s,l):
    r=np.argsort(np.argsort(s))+1; P=int(l.sum()); N=len(l)-P
    return float((r[l].sum()-P*(P+1)/2)/(P*N)) if P and N else float('nan')

def run(name, nomp, injp, provenance, TAU=0.5):
    nom=parse(nomp); inj=parse(injp); NI=min(len(nom),len(inj)); nom,inj=nom[:NI],inj[:NI]
    VP=np.zeros((NI,NC));VR=np.zeros((NI,NC));AFF=np.zeros((NI,NC),bool)
    ISIn=np.zeros((NI,NC));ISIf=np.zeros((NI,NC));DN=np.zeros((NI,NC),np.int32)
    cn=nom.sum(1).astype(np.int32);cf=inj.sum(1).astype(np.int32)
    t_vp=t_vr=t_ch=0.0
    for c in range(NC):
        for j in range(NI):
            nc=nom[j][:,(NC-1)-c]; fc=inj[j][:,(NC-1)-c]; a,b=spk(nc),spk(fc)
            AFF[j,c]=not np.array_equal(nc,fc)
            s=time.perf_counter(); VP[j,c]=victor_purpura(a,b); t_vp+=time.perf_counter()-s
            s=time.perf_counter(); VR[j,c]=van_rossum(a,b); t_vr+=time.perf_counter()-s
            s=time.perf_counter(); ISIf[j,c]=avg_isi(b); t_ch+=time.perf_counter()-s
            ISIn[j,c]=avg_isi(a); DN[j,c]=cf[j,c]-cn[j,c]
    aff=AFF.ravel();vp=VP.ravel();vr=VR.ravel();na=int(aff.sum());nu=int((~aff).sum());Nw=NI*NC
    dn=DN.ravel()
    miss=int(((dn<0)&aff).sum());spur=int(((dn>0)&aff).sum());tim=int(((dn==0)&aff).sum())
    flag=np.zeros((NI,NC),bool)
    for c in range(NC):
        m0,sd=ISIn[:,c].mean(),ISIn[:,c].std() or 1.0
        h=np.percentile(np.abs((ISIn[:,c]-m0)/sd),99); flag[:,c]=np.abs((ISIf[:,c]-m0)/sd)>h
    fl=flag.ravel();conf=vp>TAU;fu=int((fl&~aff).sum())
    res=dict(dataset=name, provenance=provenance, n_windows=Nw, n_images=NI, n_neurons=NC,
        affected_pct=round(aff.mean()*100,2), n_affected=na,
        vp_auroc=round(auroc(vp,aff),4), vr_auroc=round(auroc(vr,aff),4),
        vp_mean_affected=round(float(vp[aff].mean()),3), vp_mean_unaffected=round(float(vp[~aff].mean()),4),
        confirmation_pct=round(float((vp[aff]>TAU).mean())*100,2),
        confirmation_CI=[round(100*x,1) for x in wilson(int((vp[aff]>TAU).sum()),na)],
        false_confirmation_pct=round(float((vp[~aff]>TAU).mean())*100,3),
        false_confirmation_CI=[round(100*x,2) for x in wilson(int((vp[~aff]>TAU).sum()),nu)],
        missed_confirmation_pct=round(float(1-(vp[aff]>TAU).mean())*100,2),
        distortion_missing_pct=round(100*miss/max(1,na),1),
        distortion_spurious_pct=round(100*spur/max(1,na),1),
        distortion_timing_pct=round(100*tim/max(1,na),1),
        stage1_flagged_falsealarm=fu,
        stage2_falsealarm_rejection_pct=round(100*float((~conf&fl&~aff).sum())/max(1,fu),1),
        vp_us_per_window=round(1e6*t_vp/Nw,2), cheap_us_per_window=round(1e6*t_ch/Nw,3),
        vp_over_cheap=round(t_vp/max(1e-12,t_ch),1))
    json.dump(res, open(os.path.join(RESULTS,f"e4_stage2_{name.lower()}.json"),"w"), indent=2)
    return res

FMB=os.path.join(ART,"FashionMNIST_Experiments","Nominal"); SVB=os.path.join(ART,"SVHN_Experiments","Nominal")
cases=[
 ("MNIST", os.path.join(ART,"MNIST_Experiments","Nominal","test_spike_output_file_0.txt"),
           os.path.join(ART,"MNIST_Experiments","Output_spike_test_inject_mnist","test_spike_output_file_0.txt"),"measured (real paired outputs)"),
 ("FashionMNIST", os.path.join(FMB,"MNISTderived_nominal.txt"),
           os.path.join(ART,"FashionMNIST_Experiments","Output_spike_test_inject_Fashion_mnist","test_spike_output_file_0.txt"),"MNIST-derived"),
 ("SVHN", os.path.join(SVB,"MNISTderived_nominal.txt"),
           os.path.join(ART,"SVHN_Experiments","Output_spike_test_inject_svhn","test_spike_output_file_99.txt"),"MNIST-derived"),
]
allres=[run(*c) for c in cases]
json.dump(allres, open(os.path.join(RESULTS,"e4_stage2_all.json"),"w"), indent=2)
cols=["dataset","provenance","affected_pct","vp_auroc","vr_auroc","confirmation_pct","false_confirmation_pct","stage2_falsealarm_rejection_pct","stage1_flagged_falsealarm"]
print("\n".join([" | ".join(str(r[c]) for c in cols) for r in allres]))
print("\nwrote", os.path.join(RESULTS,"e4_stage2_all.json"))
