#!/usr/bin/env python3
"""Preprocess ClinVar TSV files for benchmarking pipeline."""
import csv
import os
import re
import sys
from collections import Counter

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

STANDARD_AA = set(AA_3TO1.values())


def is_missense(row):
    wt_3 = row['AA_Ref']
    mut_3 = row['AA_Alt']
    if wt_3 not in AA_3TO1 or mut_3 not in AA_3TO1:
        return False
    if AA_3TO1[wt_3] not in STANDARD_AA or AA_3TO1[mut_3] not in STANDARD_AA:
        return False
    return True


def clean_flanking(flanking_20kb):
    """Remove [Ref:Ref>Alt] annotation, replace with alt nucleotide."""
    return re.sub(r'\[[A-Z]:[A-Z]>([A-Z])\]', r'\1', flanking_20kb)


def get_nuc_context(flanking_20kb, ref, alt):
    """Extract 6002bp window centered on variant from 20kb flanking."""
    cleaned = clean_flanking(flanking_20kb)
    mid = len(cleaned) // 2
    start = mid - 3001
    end = mid + 3001
    if start < 0:
        start = 0
    if end > len(cleaned):
        end = len(cleaned)
    mut_context = cleaned[start:end]
    wt_context = mut_context[:mid - start] + ref + mut_context[mid - start + 1:]
    return wt_context, mut_context


def preprocess(input_path, output_dir, label_prefix=""):
    print(f"\nProcessing: {input_path}")

    variants = []
    stats = Counter()
    with open(input_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            stats['total'] += 1
            sig = r['ClinicalSignificance']
            if sig not in ('Pathogenic', 'Benign'):
                stats['skip_significance'] += 1
                continue
            if not is_missense(r):
                stats['skip_non_missense'] += 1
                continue

            wt_aa_3 = r['AA_Ref']
            mut_aa_3 = r['AA_Alt']
            wt_aa = AA_3TO1[wt_aa_3]
            mut_aa = AA_3TO1[mut_aa_3]

            try:
                pos = int(r['AA_Position'])
            except (ValueError, TypeError):
                stats['skip_bad_position'] += 1
                continue

            transcript = r['TranscriptAccession']
            gene = r['Gene']

            wt_seq = r['FullProteinSequence']
            if pos < 1 or pos > len(wt_seq):
                stats['skip_bad_seq_pos'] += 1
                continue

            flanking = r['FlankingNucleotide_20kb']
            ref_nuc = r['Ref']
            alt_nuc = r['Alt']
            nuc_wt, nuc_mut = get_nuc_context(flanking, ref_nuc, alt_nuc)

            allele_id = r['AlleleID']
            rcv = f"AL{allele_id}"

            variants.append({
                'rcv_accession': rcv,
                'gene_symbol': gene,
                'protein_position': pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'refseq_accession': transcript,
                'wt_seq': wt_seq,
                'mut_seq': wt_seq[:pos-1] + mut_aa + wt_seq[pos:],
                'clinical_significance': sig,
                'nuc_context_wt': nuc_wt,
                'nuc_context_mut': nuc_mut,
            })

    print(f"  Total raw: {stats['total']}")
    print(f"  Skip non-missense: {stats['skip_non_missense']}")
    print(f"  Skip other sig: {stats['skip_significance']}")
    print(f"  Skip bad position: {stats['skip_bad_position'] + stats['skip_bad_seq_pos']}")
    print(f"  After filtering: {len(variants)}")

    n_path = sum(1 for v in variants if v['clinical_significance'] == 'Pathogenic')
    n_ben = sum(1 for v in variants if v['clinical_significance'] == 'Benign')
    n_genes = len(set(v['gene_symbol'] for v in variants))
    print(f"  Pathogenic: {n_path}, Benign: {n_ben}, Unique genes: {n_genes}")

    os.makedirs(output_dir, exist_ok=True)

    # Remove duplicates by (gene, position, wt_aa, mut_aa) keeping first
    seen = set()
    deduped = []
    for v in variants:
        key = (v['gene_symbol'], v['protein_position'], v['wt_aa'], v['mut_aa'])
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)}")

    # Write missense_variants.tsv (protein models)
    prot_path = os.path.join(output_dir, 'missense_variants.tsv')
    fieldnames = ['rcv_accession', 'gene_symbol', 'protein_position', 'wt_aa', 'mut_aa',
                  'refseq_accession', 'wt_seq', 'mut_seq', 'clinical_significance']
    with open(prot_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for v in deduped:
            writer.writerow({k: v[k] for k in fieldnames})
    print(f"  Protein variants: {prot_path}")

    # Write dna_variants.tsv (DNA models) — extract 6002bp window
    dna_path = os.path.join(output_dir, 'dna_variants.tsv')
    dna_fieldnames = ['rcv_accession', 'gene_symbol', 'variant_position', 'nuc_context_wt',
                      'nuc_context_mut', 'label', 'wt_aa', 'mut_aa', 'protein_position']
    with open(dna_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dna_fieldnames, delimiter='\t')
        writer.writeheader()
        for v in deduped:
            mid = len(v['nuc_context_wt']) // 2
            writer.writerow({
                'rcv_accession': v['rcv_accession'],
                'gene_symbol': v['gene_symbol'],
                'variant_position': mid + 1,
                'nuc_context_wt': v['nuc_context_wt'],
                'nuc_context_mut': v['nuc_context_mut'],
                'label': v['clinical_significance'],
                'wt_aa': v['wt_aa'],
                'mut_aa': v['mut_aa'],
                'protein_position': v['protein_position'],
            })
    print(f"  DNA variants: {dna_path}")

    return len(deduped)


if __name__ == '__main__':
    v4_dir = '/ibdc-scratch2/home/Csir-igib001_lthukral/hitesh/bech_v4'

    # Dataset 1
    d1_dir = os.path.join(v4_dir, 'dataset1_clinvar_only')
    preprocess(
        os.path.join(v4_dir, 'clinvar_pathogenic_benign_only.tsv'),
        d1_dir,
    )

    # Dataset 2
    d2_dir = os.path.join(v4_dir, 'dataset2_jan2025')
    preprocess(
        os.path.join(v4_dir, 'clinvar_pathogenic_benign_jan2025.tsv'),
        d2_dir,
    )
