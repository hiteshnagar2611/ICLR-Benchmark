"""
Circularity Analysis for ICLR Paper
- D2 is a temporal subset of D1 genes (all 2569 D2 genes ∈ D1)
- Compares D1 vs D2 on shared genes (temporal validation)
- Tests whether models trained on D1-style data generalize to D2 temporal split
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICLR_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
BECH_DIR = os.path.join(ICLR_DIR, '..', 'bech_v4')

def load_dataset(ds_dir, am_path, revel_path):
    feat_path = os.path.join(ds_dir, 'feature_matrix.tsv')
    df = pd.read_csv(feat_path, sep='\t')
    df['label'] = (df['label'] == 'Pathogenic').astype(int)
    
    am = pd.read_csv(am_path, sep='\t')
    df = df.merge(am[['alphamissense_score']], left_index=True, right_index=True, how='left')
    
    rv = pd.read_csv(revel_path, sep='\t')
    df = df.merge(rv[['revel_score']], left_index=True, right_index=True, how='left')
    
    return df

def compute_auroc(labels, scores):
    mask = ~np.isnan(scores)
    if mask.sum() < 10:
        return np.nan
    return roc_auc_score(labels[mask], scores[mask])

def main():
    d1_dir = os.path.join(BECH_DIR, 'dataset1_clinvar_only')
    d2_dir = os.path.join(BECH_DIR, 'dataset2_jan2025')
    
    print("Loading D1...")
    df1 = load_dataset(d1_dir,
        os.path.join(ICLR_DIR, 'external_baselines', 'd1_alphamissense.tsv'),
        os.path.join(ICLR_DIR, 'external_baselines', 'd1_revel.tsv'))
    
    print("Loading D2...")
    df2 = load_dataset(d2_dir,
        os.path.join(ICLR_DIR, 'external_baselines', 'd2_alphamissense.tsv'),
        os.path.join(ICLR_DIR, 'external_baselines', 'd2_revel.tsv'))
    
    # D2 genes are a subset of D1 genes
    d2_genes = set(df2['gene_symbol'].unique())
    df1_d2subset = df1[df1['gene_symbol'].isin(d2_genes)]
    df1_rest = df1[~df1['gene_symbol'].isin(d2_genes)]
    
    print(f"\n=== Dataset Structure ===")
    print(f"D1 total: {len(df1)} variants, {df1['gene_symbol'].nunique()} genes")
    print(f"D2 total: {len(df2)} variants, {df2['gene_symbol'].nunique()} genes")
    print(f"D1→D2 shared genes: {len(d2_genes)} ({100*len(d2_genes)/df1['gene_symbol'].nunique():.1f}% of D1)")
    print(f"D1 on shared genes: {len(df1_d2subset)} variants")
    print(f"D1 on unique genes: {len(df1_rest)} variants")
    
    key_models = ['esm1b_mlm', 'esm1v_mlm', 'esm2_mlm', 'esm3_mlm', 'protbert_mlm', 'ntv2_mlm', 'alphamissense_score', 'revel_score']
    labels = ['ESM1b-MLM', 'ESM-1v-MLM', 'ESM2-MLM', 'ESM3-MLM', 'ProtBERT-MLM', 'NT-v2-MLM', 'AlphaMissense', 'REVEL']
    
    # Compute AUROCs for all conditions
    results = {}
    for model in key_models:
        d1_all = compute_auroc(df1['label'].values, df1[model].values)
        d1_shared = compute_auroc(df1_d2subset['label'].values, df1_d2subset[model].values)
        d1_unique = compute_auroc(df1_rest['label'].values, df1_rest[model].values)
        d2_all = compute_auroc(df2['label'].values, df2[model].values)
        results[model] = {
            'D1_all': d1_all, 'D1_shared': d1_shared, 'D1_unique': d1_unique,
            'D2_all': d2_all,
            'temporal_gap': d1_shared - d2_all
        }
        print(f"  {model:22s}: D1_all={d1_all:.4f}  D1_shared={d1_shared:.4f}  D2={d2_all:.4f}  gap={d1_shared-d2_all:+.4f}")
    
    # Save
    out_dir = os.path.join(BASE_DIR, 'circularity')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'temporal_validation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Figure: D1 vs D2 temporal validation
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(key_models))
    width = 0.3
    
    d1_all_vals = [results[m]['D1_all'] for m in key_models]
    d1_shared_vals = [results[m]['D1_shared'] for m in key_models]
    d2_vals = [results[m]['D2_all'] for m in key_models]
    
    ax.bar(x - width, d1_all_vals, width, label='D1 (all genes)', color='#2196F3', alpha=0.8)
    ax.bar(x, d1_shared_vals, width, label='D1 (shared genes only)', color='#0D47A1', alpha=0.6)
    ax.bar(x + width, d2_vals, width, label='D2 (temporal holdout)', color='#FF5722', alpha=0.8)
    
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Temporal Validation: D1 vs D2 (same gene set, different ClinVar releases)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add gap annotations
    for i, m in enumerate(key_models):
        gap = results[m]['temporal_gap']
        ax.annotate(f'{gap:+.3f}', xy=(x[i]+width, d2_vals[i]),
                   xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7, color='red')
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'temporal_validation.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir}/temporal_validation.png")
    
    # Summary
    print(f"\n=== Key Circularity Findings ===")
    print(f"D2 is a TEMPORAL holdout: all {len(d2_genes)} D2 genes exist in D1")
    print(f"But D2 has different variants from later ClinVar release (Jan 2025)")
    print(f"Temporal gaps (D1_shared - D2):")
    for m in key_models:
        gap = results[m]['temporal_gap']
        print(f"  {m:22s}: {gap:+.4f}")

if __name__ == '__main__':
    main()
