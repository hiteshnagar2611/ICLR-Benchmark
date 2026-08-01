#!/usr/bin/env python3
"""Fetch REVEL scores for our ClinVar variants by matching genomic coordinates."""

import os
import sys
import csv
import pandas as pd
import numpy as np


def load_source_variants(source_path):
    """Load original ClinVar data with genomic coordinates."""
    df = pd.read_csv(source_path, sep='\t', dtype={'Chrom': str, 'Start': int, 'Ref': str, 'Alt': str})
    return df


def load_revel_scores(revel_path):
    """Load REVEL scores indexed by chr:pos:ref:alt."""
    scores = {}
    count = 0
    with open(revel_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            chrom = row[0]
            grch38_pos = row[2]
            ref = row[3]
            alt = row[4]
            revel_score = float(row[7])
            key = f"{chrom}:{grch38_pos}:{ref}:{alt}"
            # Keep the best (highest) score if multiple transcripts
            if key not in scores or revel_score > scores[key]:
                scores[key] = revel_score
            count += 1
            if count % 10_000_000 == 0:
                print(f"  Loaded {count/1e6:.0f}M entries...")
    
    print(f"  Total REVEL entries: {count}, unique variants: {len(scores)}")
    return scores


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    am_dir = os.path.join(base_dir, 'external_baselines')
    revel_path = os.path.join(am_dir, 'revel_with_transcript_ids')
    
    datasets = [
        ('../bech_v4/clinvar_pathogenic_benign_only.tsv', '../bech_v4/dataset1_clinvar_only', 'd1'),
        ('../bech_v4/clinvar_pathogenic_benign_jan2025.tsv', '../bech_v4/dataset2_jan2025', 'd2'),
    ]
    
    # Load REVEL scores
    print("Loading REVEL scores...")
    revel_scores = load_revel_scores(revel_path)
    
    for source_rel, ds_rel, ds_name in datasets:
        source_path = os.path.abspath(os.path.join(base_dir, source_rel))
        ds_dir = os.path.abspath(os.path.join(base_dir, ds_rel))
        variants_path = os.path.join(ds_dir, 'missense_variants.tsv')
        
        if not os.path.exists(source_path) or not os.path.exists(variants_path):
            print(f"SKIP {ds_name}: source or variants not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        # Load source data with genomic coords
        source_df = load_source_variants(source_path)
        print(f"Source variants: {len(source_df)}")
        
        # Load preprocessed variants (has rcv_accession)
        proc_df = pd.read_csv(variants_path, sep='\t')
        print(f"Preprocessed variants: {len(proc_df)}")
        
        # Build genomic key for source data
        source_df['genomic_key'] = source_df['Chrom'].astype(str) + ':' + source_df['Start'].astype(str) + ':' + source_df['Ref'] + ':' + source_df['Alt']
        
        # Match source → preprocessed via rcv_accession (AlleleID)
        source_to_genomic = {}
        for _, row in source_df.iterrows():
            allele_id = 'AL' + str(row['AlleleID'])
            source_to_genomic[allele_id] = row['genomic_key']
        
        # Match REVEL scores
        scores = []
        matched = 0
        for _, row in proc_df.iterrows():
            rcv = str(row['rcv_accession'])
            if rcv in source_to_genomic:
                genomic_key = source_to_genomic[rcv]
                if genomic_key in revel_scores:
                    scores.append(revel_scores[genomic_key])
                    matched += 1
                else:
                    scores.append(np.nan)
            else:
                scores.append(np.nan)
        
        proc_df['revel_score'] = scores
        
        out_path = os.path.join(am_dir, f'{ds_name}_revel.tsv')
        proc_df[['rcv_accession', 'gene_symbol', 'protein_position', 'wt_aa', 'mut_aa',
                  'clinical_significance', 'revel_score']].to_csv(out_path, sep='\t', index=False)
        
        valid = proc_df.dropna(subset=['revel_score'])
        from sklearn.metrics import roc_auc_score
        y_true = (valid['clinical_significance'] == 'Pathogenic').astype(int)
        auroc = roc_auc_score(y_true, valid['revel_score'])
        
        print(f"Matched: {matched}/{len(proc_df)} ({100*matched/len(proc_df):.1f}%)")
        print(f"AUROC: {auroc:.4f}")
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
