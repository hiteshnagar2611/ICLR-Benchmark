#!/usr/bin/env python3
"""Download human gene→UniProt mapping and match AlphaMissense scores."""

import os
import sys
import gzip
import json
import time
import requests
import pandas as pd
import numpy as np


def download_gene_uniprot_mapping():
    """Download all human reviewed UniProt entries to build gene→UniProt mapping."""
    cache_path = os.path.join(os.path.dirname(__file__), 'external_baselines', 'human_gene_uniprot.json')
    
    if os.path.exists(cache_path):
        print(f"Loading cached mapping from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)
    
    print("Downloading human gene→UniProt mapping from UniProt...")
    url = 'https://rest.uniprot.org/uniprotkb/search'
    gene_to_uniprots = {}
    total = 0
    
    params = {
        'query': 'organism_id:9606 AND reviewed:true',
        'format': 'tsv',
        'fields': 'accession,gene_primary',
        'size': 500,
    }
    
    while True:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            print(f"  API error: {r.status_code}")
            break
        
        lines = r.text.strip().split('\n')
        for line in lines[1:]:  # Skip header
            parts = line.split('\t')
            if len(parts) >= 2:
                accession = parts[0]
                gene = parts[1]
                if gene:
                    if gene not in gene_to_uniprots:
                        gene_to_uniprots[gene] = []
                    gene_to_uniprots[gene].append(accession)
            total += 1
        
        if total % 5000 == 0:
            print(f"  Processed {total} entries, {len(gene_to_uniprots)} unique genes...")
        
        # Check for next page
        link = r.headers.get('Link', '')
        if 'rel="next"' in link:
            next_url = link.split(';')[0].strip('<>')
            url = next_url
            params = {}  # URL already contains params
            time.sleep(0.3)
        else:
            break
    
    print(f"Total: {total} entries, {len(gene_to_uniprots)} unique genes")
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(gene_to_uniprots, f)
    print(f"Saved: {cache_path}")
    
    return gene_to_uniprots


def load_am_scores(am_path, target_uniprots):
    """Load AlphaMissense scores for specific UniProt IDs."""
    scores = {}
    count = 0
    with gzip.open(am_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                uid = parts[0]
                if uid in target_uniprots:
                    variant = parts[1]
                    score = float(parts[2])
                    am_class = parts[3]
                    scores[(uid, variant)] = (score, am_class)
            count += 1
            if count % 50_000_000 == 0:
                print(f"  Scanned {count/1e6:.0f}M, matched {len(scores)}...")
    
    print(f"  Total scanned: {count}, matched: {len(scores)}")
    return scores


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    am_dir = os.path.join(base_dir, 'external_baselines')
    am_subst_path = os.path.join(am_dir, 'AlphaMissense_aa_substitutions.tsv.gz')
    
    # Step 1: Get gene→UniProt mapping
    gene_to_uniprots = download_gene_uniprot_mapping()
    
    # Collect all UniProt IDs we need
    all_uniprots = set()
    for uid_list in gene_to_uniprots.values():
        all_uniprots.update(uid_list)
    print(f"\nUnique UniProt IDs needed: {len(all_uniprots)}")
    
    # Step 2: Load AlphaMissense scores
    print("\nLoading AlphaMissense scores...")
    am_scores = load_am_scores(am_subst_path, all_uniprots)
    
    # Step 3: Process each dataset
    datasets = [
        ('../bech_v4/dataset1_clinvar_only', 'd1'),
        ('../bech_v4/dataset2_jan2025', 'd2'),
    ]
    
    for ds_rel, ds_name in datasets:
        ds_dir = os.path.abspath(os.path.join(base_dir, ds_rel))
        variants_path = os.path.join(ds_dir, 'missense_variants.tsv')
        
        if not os.path.exists(variants_path):
            print(f"SKIP {ds_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        our_vars = pd.read_csv(variants_path, sep='\t')
        print(f"Variants: {len(our_vars)}, Genes: {our_vars['gene_symbol'].nunique()}")
        
        scores = []
        classes = []
        matched = 0
        no_gene = 0
        no_variant = 0
        
        for _, row in our_vars.iterrows():
            gene = row['gene_symbol']
            variant_key = row['wt_aa'] + str(int(row['protein_position'])) + row['mut_aa']
            
            found = False
            if gene in gene_to_uniprots:
                for uid in gene_to_uniprots[gene]:
                    if (uid, variant_key) in am_scores:
                        score, am_class = am_scores[(uid, variant_key)]
                        scores.append(score)
                        classes.append(am_class)
                        matched += 1
                        found = True
                        break
                if not found:
                    no_variant += 1
            else:
                no_gene += 1
            
            if not found:
                scores.append(np.nan)
                classes.append('')
        
        our_vars['alphamissense_score'] = scores
        our_vars['alphamissense_class'] = classes
        
        out_path = os.path.join(am_dir, f'{ds_name}_alphamissense.tsv')
        our_vars[['rcv_accession', 'gene_symbol', 'protein_position', 'wt_aa', 'mut_aa',
                   'clinical_significance', 'alphamissense_score', 'alphamissense_class']].to_csv(
            out_path, sep='\t', index=False
        )
        
        valid = our_vars.dropna(subset=['alphamissense_score'])
        from sklearn.metrics import roc_auc_score
        y_true = (valid['clinical_significance'] == 'Pathogenic').astype(int)
        auroc = roc_auc_score(y_true, valid['alphamissense_score'])
        
        print(f"Matched: {matched}/{len(our_vars)} ({100*matched/len(our_vars):.1f}%)")
        print(f"No gene mapping: {no_gene}, No variant: {no_variant}")
        print(f"Score range: {np.nanmin(scores):.4f} - {np.nanmax(scores):.4f}")
        print(f"AUROC: {auroc:.4f}")
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
