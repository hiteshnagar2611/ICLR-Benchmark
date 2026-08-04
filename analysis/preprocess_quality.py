"""
Preprocess ClinVar variants with quality filtering by StarRating.
Filters to variants with >= min_stars review status.

Star ratings (ClinVar):
  3-4: reviewed by expert panel
  2: criteria provided, multiple submitters, no conflicts
  1: criteria provided, single submitter
  0: no assertion criteria provided
"""
import csv
import os
import re
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
    return True


def clean_flanking(flanking_20kb):
    return re.sub(r'\[[A-Z]:[A-Z]>([A-Z])\]', r'\1', flanking_20kb)


def get_nuc_context(flanking_20kb, ref, alt):
    cleaned = clean_flanking(flanking_20kb)
    mid = len(cleaned) // 2
    start = mid - 3001
    end = mid + 3001
    if start < 0: start = 0
    if end > len(cleaned): end = len(cleaned)
    mut_context = cleaned[start:end]
    wt_context = mut_context[:mid - start] + ref + mut_context[mid - start + 1:]
    return wt_context, mut_context


def preprocess_quality(input_path, output_dir, min_stars=2):
    print(f"\nProcessing: {input_path} (min_stars={min_stars})")

    variants = []
    stats = Counter()
    star_dist = Counter()
    review_dist = Counter()

    with open(input_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            stats['total'] += 1

            # Star rating filter
            try:
                stars = int(r.get('StarRating', 0))
            except (ValueError, TypeError):
                stars = 0
            star_dist[stars] += 1
            review_dist[r.get('ReviewStatus', 'unknown')] += 1

            if stars < min_stars:
                stats['skip_low_stars'] += 1
                continue

            sig = r['ClinicalSignificance']
            if sig not in ('Pathogenic', 'Benign'):
                stats['skip_significance'] += 1
                continue
            if not is_missense(r):
                stats['skip_non_missense'] += 1
                continue

            wt_aa = AA_3TO1[r['AA_Ref']]
            mut_aa = AA_3TO1[r['AA_Alt']]

            try:
                pos = int(r['AA_Position'])
            except (ValueError, TypeError):
                stats['skip_bad_position'] += 1
                continue

            wt_seq = r['FullProteinSequence']
            if pos < 1 or pos > len(wt_seq):
                stats['skip_bad_seq_pos'] += 1
                continue

            flanking = r['FlankingNucleotide_20kb']
            ref_nuc = r['Ref']
            alt_nuc = r['Alt']
            if not flanking or not ref_nuc or not alt_nuc:
                stats['skip_no_nuc'] += 1
                continue
            nuc_wt, nuc_mut = get_nuc_context(flanking, ref_nuc, alt_nuc)

            allele_id = r['AlleleID']
            rcv = f"AL{allele_id}"

            variants.append({
                'rcv_accession': rcv,
                'gene_symbol': r['Gene'],
                'protein_position': pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'refseq_accession': r['TranscriptAccession'],
                'wt_seq': wt_seq,
                'mut_seq': wt_seq[:pos-1] + mut_aa + wt_seq[pos:],
                'clinical_significance': sig,
                'nuc_context_wt': nuc_wt,
                'nuc_context_mut': nuc_mut,
                'star_rating': stars,
                'review_status': r.get('ReviewStatus', ''),
            })

    print(f"  Star distribution: {dict(star_dist)}")
    print(f"  Review status: {dict(review_dist)}")
    print(f"  Total raw: {stats['total']}")
    print(f"  Skip low stars: {stats['skip_low_stars']}")
    print(f"  Skip non-missense: {stats['skip_non_missense']}")
    print(f"  Skip other: {stats['skip_significance'] + stats['skip_bad_position'] + stats['skip_bad_seq_pos'] + stats.get('skip_no_nuc', 0)}")
    print(f"  After filtering: {len(variants)}")

    n_path = sum(1 for v in variants if v['clinical_significance'] == 'Pathogenic')
    n_ben = sum(1 for v in variants if v['clinical_significance'] == 'Benign')
    n_genes = len(set(v['gene_symbol'] for v in variants))
    print(f"  Pathogenic: {n_path}, Benign: {n_ben}, Genes: {n_genes}")

    # Dedup
    seen = set()
    deduped = []
    for v in variants:
        key = (v['gene_symbol'], v['protein_position'], v['wt_aa'], v['mut_aa'])
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)}")

    os.makedirs(output_dir, exist_ok=True)

    # Write protein variants
    prot_fields = ['rcv_accession', 'gene_symbol', 'protein_position', 'wt_aa', 'mut_aa',
                   'refseq_accession', 'wt_seq', 'mut_seq', 'clinical_significance',
                   'star_rating', 'review_status']
    prot_path = os.path.join(output_dir, 'missense_variants.tsv')
    with open(prot_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=prot_fields, delimiter='\t')
        writer.writeheader()
        for v in deduped:
            writer.writerow({k: v[k] for k in prot_fields})
    print(f"  Protein: {prot_path}")

    # Write DNA variants
    dna_fields = ['rcv_accession', 'gene_symbol', 'variant_position', 'nuc_context_wt',
                  'nuc_context_mut', 'label', 'wt_aa', 'mut_aa', 'protein_position',
                  'star_rating', 'review_status']
    dna_path = os.path.join(output_dir, 'dna_variants.tsv')
    with open(dna_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dna_fields, delimiter='\t')
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
                'star_rating': v['star_rating'],
                'review_status': v['review_status'],
            })
    print(f"  DNA: {dna_path}")

    return len(deduped)


if __name__ == '__main__':
    base_dir = '/ibdc-scratch2/home/Csir-igib001_lthukral/hitesh/bech_v4'
    raw_path = os.path.join(base_dir, 'clinvar_pathogenic_benign_only.tsv')

    # 2+ star filter
    n2 = preprocess_quality(raw_path, os.path.join(base_dir, 'dataset1_star2plus'), min_stars=2)
    print(f"\n=== Star 2+ dataset: {n2} variants ===")

    # 3+ star filter (expert panel only)
    n3 = preprocess_quality(raw_path, os.path.join(base_dir, 'dataset1_star3plus'), min_stars=3)
    print(f"\n=== Star 3+ dataset: {n3} variants ===")
