#!/usr/bin/env python3
"""3-fold out-of-fold uplift evaluation. Retrospective, not a new A/B test."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

POLICIES = ("uplift", "frequency", "monetary", "recency", "rfm_score")
SEED = 20261030

def load_data(ab_path, clients_path, rfm_path):
    for name in (ab_path,clients_path,rfm_path):
        if not Path(name).exists(): raise FileNotFoundError(name)
    ab=pd.read_csv(ab_path,dtype={"client_id":"string"})
    clients=pd.read_csv(clients_path,dtype={"client_id":"string"})
    rfm=pd.read_csv(rfm_path,dtype={"client_id":"string"})
    if any(not x.client_id.is_unique for x in (ab,clients,rfm)):
        raise ValueError("Duplicate client_id")
    if not ab.treatment_flg.isin([0,1]).all() or not ab.target.isin([0,1]).all():
        raise ValueError("Nonbinary treatment/target")
    d=ab.merge(clients,on="client_id",how="left",validate="one_to_one")
    d=d.merge(rfm,on="client_id",how="left",validate="one_to_one")
    if not {"count","sum","recency_days"}.issubset(d):
        raise ValueError("Need pre-campaign RFM: count, sum, recency_days")
    if d["count"].notna().mean()<.5:
        raise ValueError("Missing RFM for majority of clients; review join")
    age=pd.to_numeric(d.age,errors="coerce")
    d["age_value"]=age.where(age.between(18,100))
    d["age_unknown"]=d.age_value.isna().astype("int8")
    gender=d.gender.fillna("U").astype(str)
    for g in ("F","M","U"):
        d["gender_"+g]=(gender==g).astype("int8")
    behavior=[x for x in ("count","sum","mean_check","recency_days",
        "active_span_days","checks_30d","checks_90d","spending_90d") if x in d]
    for c in behavior:d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.reset_index(drop=True),["age_value","age_unknown","gender_F","gender_M","gender_U"]+behavior

def top(values,share=.1):
    x=np.asarray(values,float)
    if np.isnan(x).any():raise ValueError("NaN ranking")
    mask=np.zeros(len(x),bool)
    mask[np.argsort(-x,kind="stable")[:max(1,int(np.ceil(share*len(x))))]]=True
    return mask

def simple_scores(d):
    f=pd.to_numeric(d["count"],errors="coerce").fillna(-1)
    m=pd.to_numeric(d["sum"],errors="coerce").fillna(-1)
    r=pd.to_numeric(d["recency_days"],errors="coerce").fillna(np.inf)
    score=f.rank(pct=True)+m.rank(pct=True)+(-r).rank(pct=True)
    return {"frequency":f.to_numpy(float),"monetary":m.to_numpy(float),
            "recency":(-r).to_numpy(float),"rfm_score":score.to_numpy(float)}

def crossfit(d,features,folds=3,seed=SEED):
    if d.client_id.duplicated().any():raise ValueError("Duplicate client")
    X=d[features].to_numpy(float)
    t=d.treatment_flg.to_numpy(int)
    y=d.target.to_numpy(int)
    labels=2*t+y
    splits=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    oof=d[["client_id","treatment_flg","target"]].copy()
    oof["fold"]=-1
    oof["uplift_score"]=np.nan
    for policy in POLICIES:oof["select_"+policy]=False
    for k,(tr,te) in enumerate(splits.split(X,labels),1):
        score=[]
        for trt in (0,1):
            subset=tr[t[tr]==trt]
            if len(subset)<100 or len(np.unique(y[subset]))<2:
                raise ValueError("Insufficient model training labels")
            model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                HistGradientBoostingClassifier(learning_rate=.05,max_depth=4,
                    max_iter=250,min_samples_leaf=80,random_state=42))
            model.fit(X[subset],y[subset])
            score.append(model.predict_proba(X[te])[:,1])
        uplift=score[1]-score[0]
        oof.loc[te,"fold"]=k
        oof.loc[te,"uplift_score"]=uplift
        oof.loc[te,"select_uplift"]=top(uplift)
        for name,values in simple_scores(d.iloc[te]).items():
            oof.loc[te,"select_"+name]=top(values)
        print(f"Fold {k}/{folds}: trained on {len(tr):,}, evaluated on {len(te):,}")
    if oof.fold.lt(0).any() or oof.uplift_score.isna().any():
        raise AssertionError("Missing out-of-fold predictions")
    for policy in POLICIES:oof["select_"+policy]=oof["select_"+policy].astype(bool)
    return oof

def treatment_effect(oof,selected=None):
    g=oof if selected is None else oof.loc[oof[selected]]
    a=g.loc[g.treatment_flg.eq(1),"target"]
    b=g.loc[g.treatment_flg.eq(0),"target"]
    if len(a)==0 or len(b)==0:return np.nan
    return float(a.mean()-b.mean())

def reports(oof):
    global_rows=[]
    fold_rows=[]
    for fold,part in [("all",oof)]+[(str(k),d) for k,d in oof.groupby("fold")]:
        avg=treatment_effect(part)
        rows=[{"fold":fold,"policy":"random","selected":len(part),
               "uplift_pp":100*avg,"gain_vs_random_pp":0.}]
        for p in POLICIES:
            fl="select_"+p
            effect=treatment_effect(part,fl)
            rows.append({"fold":fold,"policy":p,"selected":int(part[fl].sum()),
                         "uplift_pp":100*effect,"gain_vs_random_pp":100*(effect-avg)})
        (global_rows if fold=="all" else fold_rows).extend(rows)
    return pd.DataFrame(global_rows),pd.DataFrame(fold_rows)

def paired_bootstrap(oof,repeats=2000,seed=SEED+1):
    """Exact nonparametric row bootstrap via multinomial counts over joint cells.
    Preserve policy overlap; resample independently within each OOF fold.
    Scored models are held fixed. Does not capture retraining or model selection.
    """
    if repeats<100:raise ValueError("Need >=100 bootstrap samples")
    rng=np.random.default_rng(seed)
    bins=4*2**len(POLICIES)
    accum=np.zeros((repeats,bins),dtype=np.int64)
    for _,d in oof.groupby("fold"):
        flags=np.zeros(len(d),dtype=np.int64)
        for i,name in enumerate(POLICIES):
            flags|=(d["select_"+name].to_numpy(np.int64)<<i)
        key=(d.treatment_flg.to_numpy(np.int64)*2+
             d.target.to_numpy(np.int64))*(2**len(POLICIES))+flags
        counts=np.bincount(key,minlength=bins)
        accum+=rng.multinomial(len(d),counts/len(d),size=repeats)
    idx=np.arange(bins)
    masks=idx%(2**len(POLICIES))
    t=(idx//(2**len(POLICIES)))//2
    y=(idx//(2**len(POLICIES)))%2
    def tau(select):
        treat=select&(t==1);control=select&(t==0)
        with np.errstate(divide="ignore",invalid="ignore"):
            return (accum[:,treat&(y==1)].sum(1)/accum[:,treat].sum(1)-
                    accum[:,control&(y==1)].sum(1)/accum[:,control].sum(1))
    all_tau=tau(np.ones(bins,bool))
    model_tau=tau((masks&1)!=0)
    checks={"uplift_vs_random":model_tau-all_tau}
    for bit,name in enumerate(POLICIES[1:],1):
        checks[name+"_vs_random"]=tau((masks&(1<<bit))!=0)-all_tau
    checks["uplift_vs_monetary"]=model_tau-tau((masks&(1<<2))!=0)
    checks["uplift_vs_rfm_score"]=model_tau-tau((masks&(1<<4))!=0)
    rows=[]
    for label,arr in checks.items():
        arr=arr[np.isfinite(arr)]
        if len(arr)<.99*repeats:raise ValueError("Empty treatment/control cell")
        low,high=np.quantile(arr,[.025,.975])
        rows.append({"comparison":label,
            "median_gain_pp":100*np.median(arr),
            "ci95_low_pp":100*low,"ci95_high_pp":100*high,
            "ci_contains_zero":bool(low<=0<=high)})
    return pd.DataFrame(rows)

def run(args):
    d,feat=load_data(args.ab,args.clients,args.rfm)
    print(f"{len(d):,} clients, {len(feat)} pre-campaign features")
    oof=crossfit(d,feat,folds=3,seed=SEED)
    summary,folds=reports(oof)
    intervals=paired_bootstrap(oof,repeats=args.bootstrap,seed=SEED+1)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"policy_summary.csv",index=False)
    folds.to_csv(out/"fold_results.csv",index=False)
    intervals.to_csv(out/"bootstrap_intervals.csv",index=False)
    oof.to_csv(out/"oof_scored_clients.csv",index=False)
    print("POLICY SUMMARY:\n"+summary.round(3).to_string(index=False))
    print("PAIRED CI:\n"+intervals.round(3).to_string(index=False))
    print("FOLD STABILITY:\n"+folds.round(3).to_string(index=False))
    with (out/"REPORT.md").open("w") as f:
        f.write("# Cross-fitting: historical retrospective evaluation\n\n")
        f.write("This is not a new A/B experiment or an untouched test; the dataset ")
        f.write("was previously inspected during feature/model selection. Each ")
        f.write("client's uplift score was fitted using other folds. Intervals ")
        f.write("condition on fixed scores and do not capture retraining or past ")
        f.write("model selection. Baselines: random, prespecified monetary; other ")
        f.write("RFM heuristics exploratory.\n\n")
        for name,tab in [("Policy summary",summary),("Bootstrap CI",intervals),
                         ("Per-fold stability",folds)]:
            f.write("## "+name+"\n\n"+tab.round(3).to_string(index=False)+"\n\n")
    return summary,folds,intervals

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ab",required=True)
    parser.add_argument("--clients",required=True)
    parser.add_argument("--rfm",required=True)
    parser.add_argument("--output",default="outputs/crossfit")
    parser.add_argument("--bootstrap",type=int,default=2000)
    run(parser.parse_args())
