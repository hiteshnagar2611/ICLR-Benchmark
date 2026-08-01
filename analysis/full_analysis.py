#!/usr/bin/env python3
"""
ICLR Paper: Comprehensive analysis of model agreement/disagreement.
Phase 2-4: Agreement analysis, difficulty spectrum, biological correlates, fusion ablations.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICLR_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))


MODEL_GROUPS = {
    'ESM1b': {'file': 'esm1b_mlm_results.tsv', 'group': 'Protein MLM', 'arch': 'Transformer (ESM)'},
    'ESM2': {'file': 'esm2_mlm_results.tsv', 'group': 'Protein MLM', 'arch': 'Transformer (ESM)'},
    'ESM-1v': {'file': 'esm1v_t33_650M_UR90S_1_mlm_results.tsv', 'group': 'Protein MLM', 'arch': 'Transformer (ESM)'},
    'ESM3': {'file': 'esm3_mlm_results.tsv', 'group': 'Protein MLM', 'arch': 'Transformer (ESM)'},
    'ProtBERT': {'file': 'protbert_mlm_results.tsv', 'group': 'Protein MLM', 'arch': 'Transformer (BERT)'},
    'ProtBERT-cos': {'file': 'protbert_results.tsv', 'group': 'Protein Cosine', 'arch': 'Transformer (BERT)'},
    'ESM1b-cos': {'file': 'esm1b_results.tsv', 'group': 'Protein Cosine', 'arch': 'Transformer (ESM)'},
    'ESM2-cos': {'file': 'esm2_results.tsv', 'group': 'Protein Cosine', 'arch': 'Transformer (ESM)'},
    'ESM-1v-cos': {'file': 'esm1v_results.tsv', 'group': 'Protein Cosine', 'arch': 'Transformer (ESM)'},
    'ProtT5-cos': {'file': 'prott5_results.tsv', 'group': 'Protein Cosine', 'arch': 'T5 Encoder'},
    'Ankh-cos': {'file': 'ankh_results.tsv', 'group': 'Protein Cosine', 'arch': 'T5 Encoder-Decoder'},
    'NT-v2': {'file': 'ntv2_mlm_results.tsv', 'group': 'Nucleotide MLM', 'arch': 'Transformer (ESM-based)'},
    'NT-v2-cos': {'file': 'ntv2_results.tsv', 'group': 'Nucleotide Cosine', 'arch': 'Transformer (ESM-based)'},
    'DNABERT-1': {'file': 'dnabert1_mlm_results.tsv', 'group': 'Nucleotide MLM', 'arch': 'BERT'},
    'DNABERT-2': {'file': 'dnabert2_mlm_results.tsv', 'group': 'Nucleotide MLM', 'arch': 'BERT'},
    'DNABERT-1-cos': {'file': 'dnabert1_results.tsv', 'group': 'Nucleotide Cosine', 'arch': 'BERT'},
    'DNABERT-2-cos': {'file': 'dnabert2_results.tsv', 'group': 'Nucleotide Cosine', 'arch': 'BERT'},
    'Gena-LM': {'file': 'genalm_mlm_results.tsv', 'group': 'Nucleotide MLM', 'arch': 'BERT'},
    'Gena-LM-cos': {'file': 'genalm_results.tsv', 'group': 'Nucleotide Cosine', 'arch': 'BERT'},
    'HyenaDNA': {'file': 'hyenadna_results.tsv', 'group': 'Nucleotide Cosine', 'arch': 'Hyena'},
}

EXTERNAL_MODELS = {
    'AlphaMissense': 'alphamissense_score',
    'REVEL': 'revel_score',
}

PHYS_FEATURES = [
    'grantham', 'blosum62', 'hydrophobicity_change', 'charge_change',
    'normalized_position', 'log_position', 'log_protein_length',
    'is_cysteine_change', 'is_proline_change',
]


def load_all_scores(ds_dir, am_path, revel_path):
    """Load all model scores for a dataset."""
    results_dir = os.path.join(ds_dir, 'results')
    feat_path = os.path.join(ds_dir, 'feature_matrix.tsv')

    df = pd.read_csv(feat_path, sep='\t')
    print(f"  Loaded feature_matrix: {df.shape}")

    # Add external baselines
    if am_path and os.path.exists(am_path):
        am_df = pd.read_csv(am_path, sep='\t')
        am_map = dict(zip(am_df['rcv_accession'], am_df['alphamissense_score']))
        df['AlphaMissense'] = df['rcv_accession'].map(am_map).fillna(0)
        n = df['AlphaMissense'].replace(0, np.nan).notna().sum()
        print(f"  AlphaMissense: {n}/{len(df)} matched")

    if revel_path and os.path.exists(revel_path):
        rev_df = pd.read_csv(revel_path, sep='\t')
        rev_map = dict(zip(rev_df['rcv_accession'], rev_df['revel_score']))
        df['REVEL'] = df['rcv_accession'].map(rev_map).fillna(0)
        n = df['REVEL'].replace(0, np.nan).notna().sum()
        print(f"  REVEL: {n}/{len(df)} matched")

    return df


def compute_all_aurocs(df, model_cols):
    """Compute AUROC for each model."""
    results = {}
    y_true = (df['label'] == 'Pathogenic').astype(int).values
    for col in model_cols:
        if col in df.columns:
            scores = df[col].values
            valid = ~np.isnan(scores) & (scores != 0)
            if valid.sum() > 10 and len(np.unique(y_true[valid])) > 1:
                auroc = roc_auc_score(y_true[valid], scores[valid])
                results[col] = auroc
    return results


def analyze_agreement(df, model_cols, output_dir):
    """Experiment 2.1: Model correlation structure and clustering."""
    print("\n=== Model Agreement Analysis ===")

    # Compute correlation matrix
    valid_df = df[model_cols].dropna()
    corr_matrix = valid_df.corr(method='spearman')

    print("\nSpearman correlations:")
    print(corr_matrix.round(3).to_string())

    # Hierarchical clustering
    dist_matrix = 1 - corr_matrix.abs()
    np.fill_diagonal(dist_matrix.values, 0)
    condensed = squareform(dist_matrix.values, checks=False)
    linkage_matrix = linkage(condensed, method='average')

    # Figure 1: Correlation heatmap with dendrogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 3]})

    # Dendrogram
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns.tolist(),
                       orientation='left', ax=axes[0], leaf_font_size=9)
    axes[0].set_title('Model Clustering', fontsize=13)

    # Heatmap (ordered by dendrogram)
    order = dendro['ivl']
    ordered_corr = corr_matrix.loc[order, order]

    mask = np.triu(np.ones_like(ordered_corr, dtype=bool), k=1)
    sns.heatmap(ordered_corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                square=True, ax=axes[1], cbar_kws={'label': 'Spearman r'})
    axes[1].set_title('Model Score Correlations (Spearman)', fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_correlation_heatmap.png")

    # Print cluster summary
    print("\n=== Model Groups by Correlation ===")
    protein_mlm = [c for c in model_cols if 'mlm' in c and 'ntv2' not in c and 'dna' not in c and 'genal' not in c]
    protein_cos = [c for c in model_cols if c.endswith('_cos') and 'ntv2' not in c and 'dna' not in c and 'genal' not in c and 'hyena' not in c]
    nuc_mlm = [c for c in model_cols if ('ntv2_mlm' in c or 'dnabert' in c or 'genalm' in c) and '_cos' not in c]
    nuc_cos = [c for c in model_cols if (c.endswith('_cos') and ('ntv2' in c or 'dnabert' in c or 'genal' in c or 'hyena' in c))]

    intra_protein_mlm = corr_matrix.loc[protein_mlm, protein_mlm].values
    intra_protein_mlm = intra_protein_mlm[np.triu_indices_from(intra_protein_mlm, k=1)]
    print(f"  Intra-Protein MLM: r = {np.mean(intra_protein_mlm):.3f} ± {np.std(intra_protein_mlm):.3f}")

    if nuc_mlm:
        intra_nuc_mlm = corr_matrix.loc[nuc_mlm, nuc_mlm].values
        intra_nuc_mlm = intra_nuc_mlm[np.triu_indices_from(intra_nuc_mlm, k=1)]
        print(f"  Intra-Nucleotide MLM: r = {np.mean(intra_nuc_mlm):.3f} ± {np.std(intra_nuc_mlm):.3f}")

    if protein_mlm and nuc_mlm:
        cross = corr_matrix.loc[protein_mlm, nuc_mlm].values
        print(f"  Cross-modality (Protein MLM vs Nuc MLM): r = {np.mean(cross):.3f} ± {np.std(cross):.3f}")

    return corr_matrix


def analyze_difficulty(df, model_cols, output_dir):
    """Experiment 2.2: Variant difficulty spectrum."""
    print("\n=== Difficulty Spectrum Analysis ===")

    # Compute per-variant model agreement
    scores_df = df[model_cols].copy()
    df['model_mean'] = scores_df.mean(axis=1)
    df['model_std'] = scores_df.std(axis=1)
    df['model_range'] = scores_df.max(axis=1) - scores_df.min(axis=1)

    # Stratify by difficulty (model_std)
    df['difficulty_bin'] = pd.qcut(df['model_std'], q=5, labels=['Easy', 'Easy-Med', 'Medium', 'Med-Hard', 'Hard'])

    # Compute AUROC per difficulty bin
    difficulty_results = []
    for diff_bin in ['Easy', 'Easy-Med', 'Medium', 'Med-Hard', 'Hard']:
        bin_df = df[df['difficulty_bin'] == diff_bin]
        y_true = (bin_df['label'] == 'Pathogenic').astype(int).values
        if len(np.unique(y_true)) < 2:
            continue

        result = {'difficulty': diff_bin, 'n_variants': len(bin_df)}

        for col in model_cols + ['AlphaMissense', 'REVEL']:
            if col in bin_df.columns:
                scores = bin_df[col].values
                valid = ~np.isnan(scores) & (scores != 0)
                if valid.sum() > 10:
                    result[col] = roc_auc_score(y_true[valid], scores[valid])

        difficulty_results.append(result)

    diff_df = pd.DataFrame(difficulty_results)
    print("\nAUROC by difficulty:")
    print(diff_df.to_string(index=False))

    # Figure 2: Difficulty spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUROC degradation across difficulty
    key_models = ['esm1b_mlm', 'esm1v_mlm', 'protbert_mlm', 'AlphaMissense', 'REVEL']
    key_models = [m for m in key_models if m in diff_df.columns]

    for col in key_models:
        if col in diff_df.columns:
            label = col.replace('_mlm', '').replace('_', '-')
            axes[0].plot(range(len(diff_df)), diff_df[col].values, 'o-', label=label, linewidth=2)

    axes[0].set_xticks(range(len(diff_df)))
    axes[0].set_xticklabels(diff_df['difficulty'].values, fontsize=10)
    axes[0].set_xlabel('Variant Difficulty (by model disagreement)', fontsize=11)
    axes[0].set_ylabel('AUROC', fontsize=11)
    axes[0].set_ylabel('AUROC', fontsize=11)
    axes[0].set_title('AUROC by Variant Difficulty', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    sns.despine(ax=axes[0])

    # Right: Model std distribution colored by pathogenic/benign
    path_std = df[df['label'] == 'Pathogenic']['model_std']
    ben_std = df[df['label'] == 'Benign']['model_std']
    axes[1].hist(path_std, bins=50, alpha=0.6, color='crimson', density=True, label='Pathogenic')
    axes[1].hist(ben_std, bins=50, alpha=0.6, color='steelblue', density=True, label='Benign')
    axes[1].set_xlabel('Model Disagreement (std of scores)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Model Disagreement Distribution', fontsize=13)
    axes[1].legend(fontsize=10)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_difficulty_spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig2_difficulty_spectrum.png")

    return diff_df


def analyze_biological_correlates(df, model_cols, output_dir):
    """Experiment 2.3: What biological features predict model disagreement."""
    print("\n=== Biological Feature Correlation Analysis ===")

    # Compute protein MLM vs nucleotide MLM disagreement
    prot_mlm = [c for c in ['esm1b_mlm', 'esm2_mlm', 'esm1v_mlm', 'esm3_mlm', 'protbert_mlm'] if c in df.columns]
    nuc_mlm = [c for c in ['ntv2_mlm', 'dnabert1_mlm', 'dnabert2_mlm', 'genalm_mlm'] if c in df.columns]

    if prot_mlm and nuc_mlm:
        df['prot_mlm_mean'] = df[prot_mlm].mean(axis=1)
        df['nuc_mlm_mean'] = df[nuc_mlm].mean(axis=1)
        df['cross_modal_disagreement'] = abs(df['prot_mlm_mean'] - df['nuc_mlm_mean'])

        # Correlate with biological features
        bio_features = PHYS_FEATURES + ['normalized_position', 'log_position', 'log_protein_length']
        bio_features = [f for f in bio_features if f in df.columns]

        correlations = {}
        for feat in bio_features:
            r = df['cross_modal_disagreement'].corr(df[feat], method='spearman')
            correlations[feat] = r

        corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
        print("\nCorrelation with cross-modal disagreement:")
        for feat, r in corr_series.items():
            print(f"  {feat}: r={r:.4f}")

        # Figure 3: Biological correlates
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        plot_features = ['grantham', 'blosum62', 'hydrophobicity_change',
                        'charge_change', 'normalized_position', 'log_protein_length']
        plot_features = [f for f in plot_features if f in df.columns]

        for i, feat in enumerate(plot_features[:6]):
            ax = axes[i]
            # Bin by feature value and show mean disagreement
            if df[feat].nunique() > 10:
                df['feat_bin'] = pd.qcut(df[feat], q=10, duplicates='drop')
                bin_means = df.groupby('feat_bin')['cross_modal_disagreement'].mean()
                ax.bar(range(len(bin_means)), bin_means.values, color='steelblue', edgecolor='white')
                ax.set_xticks(range(len(bin_means)))
                ax.set_xticklabels([f'{b.mid:.1f}' for b in bin_means.index], rotation=45, fontsize=7)
            else:
                group_means = df.groupby(feat)['cross_modal_disagreement'].mean()
                ax.bar(group_means.index.astype(str), group_means.values, color='steelblue', edgecolor='white')
                ax.set_xticklabels(group_means.index, rotation=45, fontsize=8)

            r = correlations.get(feat, 0)
            ax.set_xlabel(feat, fontsize=10)
            ax.set_ylabel('Cross-Modal Disagreement', fontsize=10)
            ax.set_title(f'{feat} (r={r:.3f})', fontsize=11)
            sns.despine(ax=ax)

        plt.suptitle('Biological Features vs Cross-Modal Disagreement', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig3_biological_correlates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: fig3_biological_correlates.png")

    return correlations if prot_mlm and nuc_mlm else {}


def run_fusion_ablations(df, model_cols, output_dir):
    """Phase 3: Interpretable fusion with ablation study."""
    print("\n=== Fusion Ablation Study ===")

    y = (df['label'] == 'Pathogenic').astype(int)

    # Define ablation groups
    prot_mlm_cols = [c for c in ['esm1b_mlm', 'esm2_mlm', 'esm1v_mlm', 'esm3_mlm', 'protbert_mlm'] if c in df.columns]
    nuc_mlm_cols = [c for c in ['ntv2_mlm', 'dnabert1_mlm', 'dnabert2_mlm', 'genalm_mlm'] if c in df.columns]
    prot_cos_cols = [c for c in ['esm1b_cos', 'esm2_cos', 'esm1v_cos', 'protbert_cos', 'prott5_cos', 'ankh_cos'] if c in df.columns]
    nuc_cos_cols = [c for c in ['ntv2_cos', 'dnabert1_cos', 'dnabert2_cos', 'genalm_cos', 'hyenadna_cos'] if c in df.columns]
    phys_cols = [c for c in PHYS_FEATURES if c in df.columns]
    ext_cols = [c for c in ['AlphaMissense', 'REVEL'] if c in df.columns]

    ablations = [
        ('Best Single (ESM1b-MLM)', ['esm1b_mlm']),
        ('Protein MLM Only', prot_mlm_cols),
        ('Nucleotide MLM Only', nuc_mlm_cols),
        ('Protein Cosine Only', prot_cos_cols),
        ('Nucleotide Cosine Only', nuc_cos_cols),
        ('All Protein (MLM+Cos)', prot_mlm_cols + prot_cos_cols),
        ('All Nucleotide (MLM+Cos)', nuc_mlm_cols + nuc_cos_cols),
        ('Protein + Nucleotide MLM', prot_mlm_cols + nuc_mlm_cols),
        ('All Foundation Models', prot_mlm_cols + nuc_mlm_cols + prot_cos_cols + nuc_cos_cols),
        ('Foundation + Physicochemical', prot_mlm_cols + nuc_mlm_cols + prot_cos_cols + nuc_cos_cols + phys_cols),
        ('Foundation + AlphaMissense', prot_mlm_cols + nuc_mlm_cols + ext_cols),
        ('Foundation + REVEL', prot_mlm_cols + nuc_mlm_cols + ['REVEL']),
        ('Full Ensemble (All)', prot_mlm_cols + nuc_mlm_cols + prot_cos_cols + nuc_cos_cols + phys_cols + ext_cols),
    ]

    results = []
    for name, cols in ablations:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        X = df[cols].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            verbosity=0, random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, (y_prob >= 0.5).astype(int))

        results.append({
            'Ablation': name,
            'N_features': len(cols),
            'AUROC': auroc,
            'AUPRC': auprc,
            'MCC': mcc,
        })
        print(f"  {name}: AUROC={auroc:.4f}  MCC={mcc:.4f}  ({len(cols)} features)")

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

    # Figure 4: Ablation bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(ablation_df['Ablation'], ablation_df['AUROC'], color='steelblue', edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, ablation_df['AUROC']):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('AUROC (Test Set)', fontsize=12)
    ax.set_title('Ablation Study: What Contributes to Performance?', fontsize=14)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(min(ablation_df['AUROC']) - 0.05, 1.01)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_ablation_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_ablation_study.png")

    return ablation_df


def generate_benchmark_table(df, model_cols, output_dir):
    """Generate the main benchmark comparison table."""
    print("\n=== Benchmark Table ===")

    all_models = model_cols + ['AlphaMissense', 'REVEL']
    y = (df['label'] == 'Pathogenic').astype(int)

    rows = []
    for col in all_models:
        if col in df.columns:
            scores = df[col].values
            valid = ~np.isnan(scores) & (scores != 0)
            if valid.sum() > 100:
                auroc = roc_auc_score(y[valid], scores[valid])
                auprc = average_precision_score(y[valid], scores[valid])
                rows.append({
                    'Model': col,
                    'AUROC': auroc,
                    'AUPRC': auprc,
                    'N_valid': int(valid.sum()),
                })

    table_df = pd.DataFrame(rows).sort_values('AUROC', ascending=False)
    print(table_df.to_string(index=False))
    table_df.to_csv(os.path.join(output_dir, 'benchmark_table.csv'), index=False)

    # Figure 5: Main AUROC comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    for _, row in table_df.iterrows():
        model = row['Model']
        if model == 'AlphaMissense':
            colors.append('#e63946')
        elif model == 'REVEL':
            colors.append('#f4a261')
        elif model.endswith('_mlm'):
            colors.append('#457b9d')
        elif model.endswith('_cos'):
            colors.append('#2a9d8f')
        else:
            colors.append('#888888')

    bars = ax.barh(table_df['Model'], table_df['AUROC'], color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, table_df['AUROC']):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('AUROC', fontsize=12)
    ax.set_title('Zero-Shot Pathogenicity Prediction: Comprehensive Benchmark', fontsize=14)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(min(table_df['AUROC']) - 0.05, 1.01)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_benchmark_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_benchmark_comparison.png")

    return table_df


def analyze_feature_importance(df, model_cols, output_dir):
    """SHAP-like feature importance analysis."""
    print("\n=== Feature Importance Analysis ===")

    all_feats = [c for c in model_cols + ['AlphaMissense', 'REVEL'] + PHYS_FEATURES if c in df.columns]
    y = (df['label'] == 'Pathogenic').astype(int)

    X = df[all_feats].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        subsample=0.87, colsample_bytree=0.74, min_child_weight=3,
        tree_method='hist', verbosity=0, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    importance = pd.DataFrame({
        'feature': all_feats,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\nTop 20 features:")
    for _, row in importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # Figure 6: Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance.head(20)
    colors = []
    for feat in top['feature']:
        if feat in ['AlphaMissense', 'REVEL']:
            colors.append('#e63946')
        elif feat.endswith('_mlm'):
            colors.append('#457b9d')
        elif feat.endswith('_cos'):
            colors.append('#2a9d8f')
        elif feat in PHYS_FEATURES:
            colors.append('#f4a261')
        else:
            colors.append('#888888')

    ax.barh(range(len(top)), top['importance'].values[::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance (XGBoost Gain)', fontsize=12)
    ax.set_title('Feature Importance for Variant Pathogenicity Prediction', fontsize=13)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig6_feature_importance.png")

    return importance


def main():
    ds_dirs = {
        'd1': {
            'ds_dir': os.path.join(ICLR_DIR, '..', 'bech_v4', 'dataset1_clinvar_only'),
            'am_path': os.path.join(BASE_DIR, '..', 'external_baselines', 'd1_alphamissense.tsv'),
            'revel_path': os.path.join(BASE_DIR, '..', 'external_baselines', 'd1_revel.tsv'),
        },
        'd2': {
            'ds_dir': os.path.join(ICLR_DIR, '..', 'bech_v4', 'dataset2_jan2025'),
            'am_path': os.path.join(BASE_DIR, '..', 'external_baselines', 'd2_alphamissense.tsv'),
            'revel_path': os.path.join(BASE_DIR, '..', 'external_baselines', 'd2_revel.tsv'),
        },
    }

    for ds_name, ds_info in ds_dirs.items():
        output_dir = os.path.join(BASE_DIR, 'analysis', ds_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*60}")

        df = load_all_scores(ds_info['ds_dir'], ds_info['am_path'], ds_info['revel_path'])

        # Get model columns (all our foundation model scores)
        model_cols = [c for c in df.columns if c.endswith('_mlm') or c.endswith('_cos')]
        model_cols = [c for c in model_cols if c not in ('mlm_mean', 'mlm_std', 'mlm_max', 'mlm_min',
                                                          'mlm_esm1b_esm2_diff', 'mlm_esm1b_protbert_diff',
                                                          'cos_mean', 'cos_std')]

        # Run analyses
        corr_matrix = analyze_agreement(df, model_cols, output_dir)
        diff_df = analyze_difficulty(df, model_cols, output_dir)
        bio_corr = analyze_biological_correlates(df, model_cols, output_dir)
        ablation_df = run_fusion_ablations(df, model_cols, output_dir)
        bench_table = generate_benchmark_table(df, model_cols, output_dir)
        importance = analyze_feature_importance(df, model_cols, output_dir)

        # Save summary
        summary = {
            'dataset': ds_name,
            'n_variants': len(df),
            'n_genes': df['gene_symbol'].nunique(),
            'n_models': len(model_cols),
            'benchmark_table': bench_table.to_dict('records'),
            'ablation_results': ablation_df.to_dict('records'),
        }
        with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nAll analysis saved to {output_dir}/")


if __name__ == '__main__':
    main()
