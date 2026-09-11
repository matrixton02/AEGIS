"""
load_smd.py

Loads the real Server Machine Dataset (SMD, from the OmniAnomaly paper,
NetManAIOps/OmniAnomaly on GitHub) and runs the same correlation /
autocorrelation EDA we ran on placeholder data -- except now grounded
in real production server metrics (CPU, memory, network, etc., though
column names are anonymized in the public release).

Directory layout expected (already downloaded via raw.githubusercontent.com):
  smd/train/<machine>.txt       -- assumed anomaly-free, used for training
  smd/test/<machine>.txt        -- same 38 columns, contains real anomalies
  smd/test_label/<machine>.txt  -- one 0/1 label per row of test/<machine>.txt

Each .txt file is headerless, comma-separated, already min-max normalized
to [0, 1] by the dataset authors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

SMD_DIR="SMD"
MACHINES = ["machine-1-1", "machine-1-2", "machine-2-1"]
N_FEATURES=38
FEATURE_COLS = [f"f{i:02d}" for i in range(N_FEATURES)]
SAMPLE_INTERVAL_SEC = 60   # SMD is documented as 1-minute granularity in the OmniAnomaly paper

def load_split(split: str, machines: list[str]=MACHINES)->pd.DataFrame:
    frames=[]
    for m in machines:
        path=f"{SMD_DIR}/{split}/{m}.txt"
        df=pd.read_csv(path,header=None,names=FEATURE_COLS)
        df["machine_id"]=m
        df["time_stamp"]=np.arange(len(df))* SAMPLE_INTERVAL_SEC

        label_path=f"{SMD_DIR}/test_label/{m}.txt"
        if split=="test" and os.path.exists(label_path):
            lables=pd.read_csv(label_path,header=None,names=["is_anomaly"])
            df["is_anomaly"]=lables["is_anomaly"].to_numpy()
        else:
            df["is_anomaly"]=0 # train split is assumed anomaly-free by SMD's own protocol

        frames.append(df)
    return pd.concat(frames,ignore_index=True)

def summarize_and_correlate(df: pd.DataFrame, outpath: str)->pd.DataFrame:
    corr=df[FEATURE_COLS].corr()
    print(f"\n{len(FEATURE_COLS)} anonymized metrics -- showing correlation strength distribution:")
    off_diag=corr.to_numpy()[~np.eye(len(FEATURE_COLS),dtype=bool)]
    n_constant=df[FEATURE_COLS].std().eq(0).sum()
    if n_constant:
         print(f"  note: {n_constant} of {len(FEATURE_COLS)} metrics are constant on this machine subset "
               f"(zero variance -> undefined correlation, excluded below)")
         
    print(f"  mean |correlation| across all metric pairs: {np.nanmean(np.abs(off_diag)):.3f}")
    print(f"  fraction of pairs with |corr| > 0.5: {np.nanmean(np.abs(off_diag) > 0.5)*100:.1f}%")

    plt.figure(figsize=(11,9))
    sns.heatmap(corr,cmap="coolwarm",vmin=-1,vmax=1,square=True,xticklabels=True,yticklabels=True,cbar_kws={"labels":"correlations"})
    plt.title("SMD: Correlation across 38 anynymized metrics")
    plt.tight_layout()
    plt.savefig(outpath,dpi=150)
    print(f"Sabed SMD corelation heatmap-> {outpath}")
    return corr

if __name__=="__main__":
    out_dir="EDA_Output"
    os.makedirs(out_dir,exist_ok=True)
