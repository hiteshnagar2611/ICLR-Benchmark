# ICLR 2027 Paper Plan

**Title (working):** "Representation Geometry of Protein and Nucleotide Foundation Models for Variant Pathogenicity: A Controlled Multi-Modal Benchmark"

**Deadline:** Abstract September 19, 2026 | Paper September 24, 2026
**Conference:** April 24-28, 2027, Brazil

---

## Paper Narrative

We systematically benchmark **9 protein** + **4 nucleotide** foundation models on **29,595 ClinVar variants** using two scoring paradigms (MLM + cosine), compare against **AlphaMissense/REVEL/EVE/CADD**, and discover that:

1. **Protein MLM models capture a shared signal** (r = 0.68-0.83) while **nucleotide models capture orthogonal information** (r ≈ 0.00)
2. **Model disagreement patterns correlate with biology**: variants in disordered regions, near splice sites, or at domain boundaries show higher disagreement
3. **A lightweight fusion** of protein + nucleotide scores matches AlphaMissense performance (AUROC 0.94-0.95) **without population frequency data**

---

## Current Results (Baseline)

### D1 (30K variants) Test Set AUROC
| Model | AUROC |
|---|---|
| REVEL | 0.9607 |
| AlphaMissense | 0.9475 |
| Meta-Learner (Ours) | 0.9378 |
| ESM1b-MLM | 0.8860 |
| ESM-1v-MLM | 0.8736 |
| ProtBERT-MLM | 0.8482 |
| ESM2-MLM | 0.8479 |
| ESM3-MLM | 0.8297 |
| ProtBERT (cosine) | 0.7625 |
| NT-v2-MLM | 0.6374 |

### D2 (10K variants) Test Set AUROC
| Model | AUROC |
|---|---|
| REVEL | 0.9732 |
| AlphaMissense | 0.9517 |
| Meta-Learner (Ours) | 0.9489 |
| ESM1b-MLM | 0.9033 |
| ESM-1v-MLM | 0.8881 |
| ProtBERT-MLM | 0.8624 |
| ESM2-MLM | 0.8640 |
| ESM3-MLM | 0.8479 |

### Ablation Study (D1)
| Configuration | AUROC | MCC |
|---|---|---|
| Full Ensemble (All) | 0.9748 | 0.855 |
| Foundation + AlphaMissense | 0.9695 | 0.831 |
| Foundation + REVEL | 0.9618 | 0.807 |
| Foundation + Physicochemical | 0.9370 | 0.737 |
| All Foundation Models | 0.9266 | 0.706 |
| All Protein (MLM+Cos) | 0.9225 | 0.701 |
| Best Single (ESM1b-MLM) | 0.8771 | 0.604 |

### Ablation Study (D2)
| Configuration | AUROC | MCC |
|---|---|---|
| Full Ensemble (All) | 0.9868 | 0.899 |
| Foundation + AlphaMissense | 0.9822 | 0.879 |
| Foundation + REVEL | 0.9788 | 0.862 |
| Foundation + Physicochemical | 0.9509 | 0.755 |
| All Foundation Models | 0.9375 | 0.723 |
| All Protein (MLM+Cos) | 0.9334 | 0.724 |
| Best Single (ESM1b-MLM) | 0.9021 | 0.672 |

### Key Correlation Findings
- Protein MLM models: r = 0.68-0.83 (highly correlated)
- Nucleotide models vs protein models: r ≈ 0.00 (orthogonal!)
- NT-v2: r = 0.15-0.29 (weak bridge)
- Intra-Protein MLM: r = 0.393 ± 0.366
- Intra-Nucleotide MLM: r = 0.071 ± 0.065
- Cross-modality: r = 0.059 ± 0.120

### Circularity / Temporal Validation
- D2 is a temporal holdout: all 2,569 D2 genes exist in D1
- Temporal gaps (D1 - D2) are small (<0.015 for foundation models)
- Foundation models perform slightly BETTER on D2 temporal test
- AlphaMissense gap: -0.0047 (nearly identical)

---

## Required Experiments (7 weeks)

### Phase 1: External Baseline Scores (Week 1-2)

| Source | What to Download | How to Match |
|---|---|---|
| **AlphaMissense** | `AlphaMissense_aa_substitutions.tsv.gz` from Zenodo (8208688) | Match by gene + AA change (e.g., "R330M") |
| **REVEL** | `revel-v1.3_all_chromosomes.zip` from Zenodo (7072866) | Match by GRCh38 genomic coordinates |
| **CADD** | `whole_genome_SNVs.tsv.gz` from CADD website | Match by genomic coordinates (tabix query) |
| **EVE** | Bulk download from evemodel.org | Match by gene symbol + AA position |

**Matching strategy:** Our variants have `gene_symbol`, `protein_position`, `wt_aa`, `mut_aa`. For AlphaMissense we match on gene + wt_aa + position + mut_aa. For REVEL/CADD we need genomic coordinates.

**Scripts needed:**
- `fetch_alpha_missense.py` — Download + match
- `fetch_revel_cadd.py` — Download + match via genomic coordinates
- `fetch_eve.py` — Download + match

### Phase 2: Agreement/Disagreement Analysis (Week 2-4)

**Experiment 2.1: Model Correlation Structure**
- Pairwise Pearson/Spearman correlations between all 22 model scores
- Hierarchical clustering of models (protein vs nucleotide vs external)
- UMAP/t-SNE visualization of variant score space
- **Finding to present:** Protein models cluster together; nucleotide models are orthogonal

**Experiment 2.2: Variant Difficulty Spectrum**
- For each variant, compute "model agreement score" = std dev of all model predictions
- Stratify variants into easy (low disagreement) vs hard (high disagreement)
- Compare AUROC on easy vs hard subsets for each model
- **Finding to present:** Nucleotide models add most value on "hard" variants where protein models disagree

**Experiment 2.3: Biological Feature Correlation**
- For variants where protein vs nucleotide models disagree:
  - Check if they're near intrinsically disordered regions (use pLDDT proxy)
  - Check if they're near splice junctions
  - Check amino acid properties (Grantham distance, charge change)
  - Check protein length and position
- **Finding to present:** Disagreement correlates with disorder, splice proximity, and extreme physicochemical changes

**Experiment 2.4: Architecture Analysis**
- Group models by architecture (ESM family, ProtBERT, T5-based, DNA transformers)
- Compare intra-group vs inter-group agreement
- Analyze effect of model size (ESM1b 650M vs ESM2 3B vs ESM3 1.4B)
- **Finding to present:** Architecture matters more than size for variant prediction

### Phase 3: Meta-Learner with Interpretable Fusion (Week 4-5)

Instead of black-box XGBoost, build an **interpretable fusion**:

1. **Linear combination baseline**: Learn weights for each model score
2. **Attention-based fusion**: Simple attention layer that weights models per-variant
3. **SHAP analysis**: Which models matter most for which variant types?

**Key ablations:**
- Protein models only → AUROC X
- Nucleotide models only → AUROC Y
- Protein + Nucleotide → AUROC Z (expect Z > max(X, Y))
- All + physicochemical features → final AUROC

### Phase 4: Circularity Analysis (Week 5-6)

1. **Gene overlap analysis**: Split test variants into genes present vs absent from each model's training data
2. **Temporal validation**: Use D2 (Jan 2025 ClinVar) as independent temporal test set
3. **Compare zero-shot vs trained**: Show that our meta-learner trained only on protein model outputs (no population data) matches AlphaMissense (trained on population frequencies)

### Phase 5: Writing (Week 6-7)

**Paper structure:**
1. Introduction: Why variant prediction matters, gap in controlled benchmarks
2. Related Work: Foundation models for biology, variant effect prediction
3. Methods: Dataset construction, model descriptions, scoring protocols, meta-learner
4. Experiments:
   - 4.1 Benchmark Results (table with all models + external baselines)
   - 4.2 Model Agreement Analysis (correlation structure, clustering)
   - 4.3 What Makes Variants Hard? (difficulty spectrum, biological correlates)
   - 4.4 Multi-Modal Fusion (ablation study, interpretable fusion)
   - 4.5 Circularity Analysis (gene overlap, temporal validation)
5. Discussion: Implications for variant prediction, when to use which model
6. Conclusion

---

## Key Figures Needed

1. **Figure 1**: Model correlation heatmap (protein vs nucleotide cluster structure)
2. **Figure 2**: AUROC comparison bar chart (all models + AlphaMissense/REVEL)
3. **Figure 3**: Variant difficulty spectrum (easy → hard, model performance degradation)
4. **Figure 4**: Biological feature correlation (disagreement vs disorder, charge, position)
5. **Figure 5**: Ablation study (protein only → nucleotide only → fusion → AlphaMissense)
6. **Figure 6**: UMAP of model score space (colored by pathogenic/benign)

---

## Data Pipeline

```
Existing data (source: bech_v4/):
  dataset1_clinvar_only/results/*.tsv  (22 model scores, 29595 variants)
  dataset2_jan2025/results/*.tsv       (22 model scores, 9627 variants)
  dataset1_clinvar_only/feature_matrix.tsv
  dataset1_clinvar_only/meta_model/
  dataset2_jan2025/feature_matrix.tsv
  dataset2_jan2025/meta_model/

New data (to fetch):
  external_baselines/alphamissense_scores.tsv
  external_baselines/revel_scores.tsv
  external_baselines/cadd_scores.tsv
  external_baselines/eve_scores.tsv

New analysis scripts (analysis/):
  agreement_analysis.py       — Model correlation structure + clustering
  difficulty_spectrum.py      — Easy vs hard variant analysis
  biological_correlates.py    — Disagreement vs biology features
  interpretable_fusion.py     — Linear/attention fusion + ablations
  circularity_analysis.py     — Gene overlap + temporal validation
  generate_figures.py         — All paper figures
  fetch_alpha_missense.py     — Download + match AlphaMissense
  fetch_revel_cadd.py         — Download + match REVEL/CADD
  fetch_eve.py                — Download + match EVE
```

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| AlphaMissense matching fails | Use dbNSFP as intermediary |
| REVEL coordinate matching noisy | Validate with gene:AA; filter clean matches only |
| EVE doesn't cover all genes | Report coverage; evaluate on subset |
| CADD file is 80GB | Use tabix to query only our positions |
| 7 weeks too tight | Prioritize: baselines (wk1-2), agreement (wk2-3), figures (wk3-4), writing (wk5-7) |
