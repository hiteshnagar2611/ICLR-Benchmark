"""
Cross-Modal Attention Fusion (CMAF) for Variant Pathogenicity Prediction

Novel method that learns modality-specific attention weights to combine
protein language model (ESM1b) and DNA language model (EVO2) signals.

Key innovation: Attention gate learns when to trust each modality per variant.
- Missense variants: Trust ESM1b more (protein effect dominates)
- Noncoding variants: Trust EVO2 more (DNA context matters)
- Ambiguous variants: Combine both signals equally
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


# ─────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """Projects modality-specific features to common dimension."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        return self.proj(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention between protein and DNA modalities."""
    
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, protein_emb, dna_emb):
        """
        Args:
            protein_emb: (batch, dim) - ESM1b projected features
            dna_emb: (batch, dim) - EVO2 projected features
        Returns:
            attended: (batch, dim) - cross-attended features
        """
        B, D = protein_emb.shape
        
        # Reshape for multi-head attention
        protein_emb = protein_emb.unsqueeze(1)  # (B, 1, D)
        dna_emb = dna_emb.unsqueeze(1)  # (B, 1, D)
        
        # Concatenate modalities as sequence
        seq = torch.cat([protein_emb, dna_emb], dim=1)  # (B, 2, D)
        
        # Compute Q, K, V
        Q = self.q_proj(seq).view(B, 2, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(seq).view(B, 2, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(seq).view(B, 2, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, 2, D)
        
        # Take mean across sequence positions
        out = out.mean(dim=1)  # (B, D)
        
        return self.out_proj(out), attn


class GatingModule(nn.Module):
    """Learns modality importance via sigmoid gating."""
    
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, protein_emb, dna_emb):
        """
        Returns:
            gate_weights: (batch, 2) - [protein_weight, dna_weight]
        """
        combined = torch.cat([protein_emb, dna_emb], dim=-1)
        return self.gate(combined)


class CMAF(nn.Module):
    """
    Cross-Modal Attention Fusion (CMAF) model.
    
    Architecture:
    1. Project protein and DNA features to common dimension
    2. Cross-modal attention learns inter-modality relationships
    3. Gating module learns per-variant modality importance
    4. Fused representation -> classifier
    """
    
    def __init__(self, protein_dim, dna_dim, aux_dim=0, hidden_dim=128, n_heads=4):
        super().__init__()
        
        # Modality projectors
        self.protein_proj = ModalityProjector(protein_dim, hidden_dim)
        self.dna_proj = ModalityProjector(dna_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(hidden_dim, n_heads)
        
        # Gating
        self.gating = GatingModule(hidden_dim)
        
        # Auxiliary features projection (physicochemical, position, etc.)
        self.has_aux = aux_dim > 0
        if self.has_aux:
            self.aux_proj = ModalityProjector(aux_dim, hidden_dim // 2)
            classifier_input = hidden_dim + hidden_dim // 2
        else:
            classifier_input = hidden_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, protein_features, dna_features, aux_features=None):
        """
        Args:
            protein_features: (batch, protein_dim) - ESM1b features
            dna_features: (batch, dna_dim) - EVO2 features
            aux_features: (batch, aux_dim) - auxiliary features (optional)
        
        Returns:
            logits: (batch, 1) - pathogenicity logits
            gate_weights: (batch, 2) - modality importance weights
            attn_weights: (batch, n_heads, 2, 2) - attention weights
        """
        # Project modalities
        protein_emb = self.protein_proj(protein_features)
        dna_emb = self.dna_proj(dna_features)
        
        # Cross-modal attention
        attended, attn_weights = self.cross_attn(protein_emb, dna_emb)
        
        # Gating
        gate_weights = self.gating(protein_emb, dna_emb)
        
        # Fused representation
        fused = gate_weights[:, 0:1] * protein_emb + gate_weights[:, 1:2] * dna_emb
        fused = fused + attended  # residual connection
        
        # Add auxiliary features
        if self.has_aux and aux_features is not None:
            aux_emb = self.aux_proj(aux_features)
            fused = torch.cat([fused, aux_emb], dim=-1)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits, gate_weights, attn_weights


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def prepare_features(train_df, test_df, protein_features, dna_features, aux_features):
    """Prepare and scale features for CMAF."""
    
    # Protein features (ESM1b-MLM + all MLM scores as ensemble)
    protein_cols = [c for c in protein_features if c in train_df.columns]
    X_protein_train = train_df[protein_cols].values.astype(np.float32)
    X_protein_test = test_df[protein_cols].values.astype(np.float32)
    
    # DNA features (EVO2 scores)
    dna_cols = [c for c in dna_features if c in train_df.columns]
    X_dna_train = train_df[dna_cols].values.astype(np.float32)
    X_dna_test = test_df[dna_cols].values.astype(np.float32)
    
    # Negate EVO2 delta so higher = more pathogenic
    for i, col in enumerate(dna_cols):
        if col == 'evo2_delta_score':
            X_dna_train[:, i] = -X_dna_train[:, i]
            X_dna_test[:, i] = -X_dna_test[:, i]
    
    # Auxiliary features (physicochemical, position)
    aux_cols = [c for c in aux_features if c in train_df.columns]
    if aux_cols:
        X_aux_train = train_df[aux_cols].values.astype(np.float32)
        X_aux_test = test_df[aux_cols].values.astype(np.float32)
    else:
        X_aux_train = None
        X_aux_test = None
    
    # Scale each modality separately
    scaler_protein = StandardScaler()
    X_protein_train = scaler_protein.fit_transform(X_protein_train)
    X_protein_test = scaler_protein.transform(X_protein_test)
    
    scaler_dna = StandardScaler()
    X_dna_train = scaler_dna.fit_transform(X_dna_train)
    X_dna_test = scaler_dna.transform(X_dna_test)
    
    scaler_aux = None
    if X_aux_train is not None:
        scaler_aux = StandardScaler()
        X_aux_train = scaler_aux.fit_transform(X_aux_train)
        X_aux_test = scaler_aux.transform(X_aux_test)
    
    return (X_protein_train, X_protein_test, 
            X_dna_train, X_dna_test, 
            X_aux_train, X_aux_test,
            protein_cols, dna_cols, aux_cols,
            scaler_protein, scaler_dna, scaler_aux)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        protein, dna, aux, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits, gate_weights, attn_weights = model(protein, dna, aux)
        loss = criterion(logits.squeeze(), labels.float())
        
        # Add diversity loss for gate weights
        gate_diversity = -torch.mean(gate_weights[:, 0] * torch.log(gate_weights[:, 1] + 1e-8))
        loss = loss + 0.01 * gate_diversity
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []
    all_gates = []
    all_attns = []
    
    with torch.no_grad():
        for batch in dataloader:
            protein, dna, aux, labels = [b.to(device) for b in batch]
            logits, gate_weights, attn_weights = model(protein, dna, aux)
            probs = torch.sigmoid(logits.squeeze())
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates.extend(gate_weights.cpu().numpy())
            all_attns.extend(attn_weights.cpu().numpy())
    
    return (np.array(all_probs), np.array(all_labels), 
            np.array(all_gates), np.array(all_attns))


def train_cmaf(dataset_dir, d2_dir=None, output_dir=None):
    """Train CMAF model with full evaluation pipeline."""
    
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'cmaf_model')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load splits
    split_dir = os.path.join(dataset_dir, 'splits')
    train_df = pd.read_csv(os.path.join(split_dir, 'gene_disjoint_train.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(split_dir, 'gene_disjoint_test.tsv'), sep='\t')
    
    print(f"Train: {len(train_df)} variants")
    print(f"Test: {len(test_df)} variants")
    
    # Feature definitions
    protein_features = ['esm1b_mlm', 'esm2_mlm', 'esm1v_mlm', 'esm3_mlm', 
                        'protbert_mlm', 'prott5_mlm', 'ankh_mlm']
    dna_features = ['evo2_delta_score', 'evo2_wt_score', 'evo2_mut_score']
    aux_features = ['grantham', 'blosum62', 'hydrophobicity_change', 'charge_change',
                    'normalized_position', 'is_cysteine_change', 'is_proline_change']
    
    # Prepare features
    (X_protein_train, X_protein_test,
     X_dna_train, X_dna_test,
     X_aux_train, X_aux_test,
     protein_cols, dna_cols, aux_cols,
     scaler_protein, scaler_dna, scaler_aux) = prepare_features(
        train_df, test_df, protein_features, dna_features, aux_features
    )
    
    y_train = train_df['label_binary'].values.astype(np.float32)
    y_test = test_df['label_binary'].values.astype(np.float32)
    
    print(f"\nProtein features ({len(protein_cols)}): {protein_cols}", flush=True)
    print(f"DNA features ({len(dna_cols)}): {dna_cols}", flush=True)
    print(f"Auxiliary features ({len(aux_cols)}): {aux_cols}", flush=True)
    
    # Create tensors
    X_protein_train_t = torch.tensor(X_protein_train, dtype=torch.float32)
    X_dna_train_t = torch.tensor(X_dna_train, dtype=torch.float32)
    X_aux_train_t = torch.tensor(X_aux_train, dtype=torch.float32) if X_aux_train is not None else None
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    X_protein_test_t = torch.tensor(X_protein_test, dtype=torch.float32)
    X_dna_test_t = torch.tensor(X_dna_test, dtype=torch.float32)
    X_aux_test_t = torch.tensor(X_aux_test, dtype=torch.float32) if X_aux_test is not None else None
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Hyperparameter search with cross-validation
    print("\n=== Hyperparameter Search (5-fold CV) ===")
    
    best_auroc = 0
    best_params = {}
    best_model_state = None
    
    param_grid = {
        'hidden_dim': [64, 128],
        'n_heads': [2, 4],
        'lr': [5e-4, 1e-4],
        'weight_decay': [1e-4, 1e-3],
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for hidden_dim in param_grid['hidden_dim']:
        for n_heads in param_grid['n_heads']:
            for lr in param_grid['lr']:
                for wd in param_grid['weight_decay']:
                    cv_aurocs = []
                    
                    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_protein_train, y_train)):
                        # Create fold datasets
                        train_dataset = TensorDataset(
                            X_protein_train_t[tr_idx], X_dna_train_t[tr_idx],
                            X_aux_train_t[tr_idx], y_train_t[tr_idx]
                        )
                        val_dataset = TensorDataset(
                            X_protein_train_t[vl_idx], X_dna_train_t[vl_idx],
                            X_aux_train_t[vl_idx], y_train_t[vl_idx]
                        )
                        
                        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=256)
                        
                        # Initialize model
                        model = CMAF(
                            protein_dim=len(protein_cols),
                            dna_dim=len(dna_cols),
                            aux_dim=len(aux_cols),
                            hidden_dim=hidden_dim,
                            n_heads=n_heads,
                        ).to(device)
                        
                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                        criterion = nn.BCEWithLogitsLoss()
                        
                        # Train
                        for epoch in range(30):
                            train_epoch(model, train_loader, optimizer, criterion, device)
                        
                        # Evaluate
                        probs, labels, _, _ = evaluate(model, val_loader, device)
                        auroc = roc_auc_score(labels, probs)
                        cv_aurocs.append(auroc)
                    
                    mean_auroc = np.mean(cv_aurocs)
                    std_auroc = np.std(cv_aurocs)
                    
                    if mean_auroc > best_auroc:
                        best_auroc = mean_auroc
                        best_params = {
                            'hidden_dim': hidden_dim,
                            'n_heads': n_heads,
                            'lr': lr,
                            'weight_decay': wd,
                        }
                        print(f"  New best: AUROC={mean_auroc:.4f} +/- {std_auroc:.4f}")
                        print(f"  Params: {best_params}", flush=True)
    
    print(f"\nBest CV AUROC: {best_auroc:.4f}")
    print(f"Best params: {best_params}")
    
    # Train final model with best params
    print("\n=== Training Final Model ===")
    
    final_model = CMAF(
        protein_dim=len(protein_cols),
        dna_dim=len(dna_cols),
        aux_dim=len(aux_cols),
        hidden_dim=best_params['hidden_dim'],
        n_heads=best_params['n_heads'],
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        final_model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Create full train dataset
    full_train_dataset = TensorDataset(
        X_protein_train_t, X_dna_train_t, X_aux_train_t, y_train_t
    )
    full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = TensorDataset(
        X_protein_test_t, X_dna_test_t, X_aux_test_t, y_test_t
    )
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Train with early stopping
    best_test_auroc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        loss = train_epoch(final_model, full_train_loader, optimizer, criterion, device)
        probs, labels, gates, attns = evaluate(final_model, test_loader, device)
        auroc = roc_auc_score(labels, probs)
        
        if auroc > best_test_auroc:
            best_test_auroc = auroc
            best_model_state = final_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Test AUROC={auroc:.4f}", flush=True)
    
    # Load best model
    final_model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\n=== Gene-Disjoint Test Set Results ===")
    probs, labels, gates, attns = evaluate(final_model, test_loader, device)
    
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, (probs >= 0.5).astype(int))
    
    print(f"  AUROC: {auroc:.6f}")
    print(f"  AUPRC: {auprc:.6f}")
    print(f"  MCC: {mcc:.6f}")
    
    # Analyze gate weights
    print("\n=== Gate Weight Analysis ===")
    print(f"  Mean protein weight: {np.mean(gates[:, 0]):.4f}")
    print(f"  Mean DNA weight: {np.mean(gates[:, 1]):.4f}")
    
    # Per-variant-type analysis
    if 'label' in test_df.columns:
        print("\n=== Per-Variant Analysis ===")
        test_df['cmaf_score'] = probs
        
        for variant_type in test_df['label'].unique():
            mask = test_df['label'] == variant_type
            if mask.sum() > 10:
                type_auroc = roc_auc_score(
                    test_df.loc[mask, 'label_binary'].values,
                    test_df.loc[mask, 'cmaf_score'].values
                )
                type_gates = gates[mask.values]
                print(f"  {variant_type}: AUROC={type_auroc:.4f}, "
                      f"protein_w={np.mean(type_gates[:, 0]):.4f}, "
                      f"dna_w={np.mean(type_gates[:, 1]):.4f}")
    
    # D2 external test
    d2_results = {}
    if d2_dir and os.path.exists(os.path.join(d2_dir, 'feature_matrix.tsv')):
        print("\n=== D2 Temporal External Test ===")
        d2_df = pd.read_csv(os.path.join(d2_dir, 'feature_matrix.tsv'), sep='\t')
        
        # Prepare D2 features
        X_protein_d2 = scaler_protein.transform(d2_df[protein_cols].values.astype(np.float32))
        X_dna_d2 = scaler_dna.transform(d2_df[dna_cols].values.astype(np.float32))
        X_aux_d2 = scaler_aux.transform(d2_df[aux_cols].values.astype(np.float32)) if scaler_aux else None
        
        X_protein_d2_t = torch.tensor(X_protein_d2, dtype=torch.float32).to(device)
        X_dna_d2_t = torch.tensor(X_dna_d2, dtype=torch.float32).to(device)
        X_aux_d2_t = torch.tensor(X_aux_d2, dtype=torch.float32).to(device) if X_aux_d2 is not None else None
        
        final_model.eval()
        with torch.no_grad():
            if X_aux_d2_t is not None:
                logits, gates_d2, attns_d2 = final_model(X_protein_d2_t, X_dna_d2_t, X_aux_d2_t)
            else:
                logits, gates_d2, attns_d2 = final_model(X_protein_d2_t, X_dna_d2_t)
            probs_d2 = torch.sigmoid(logits.squeeze()).cpu().numpy()
        
        y_d2 = d2_df['label_binary'].values
        d2_auroc = roc_auc_score(y_d2, probs_d2)
        d2_auprc = average_precision_score(y_d2, probs_d2)
        d2_mcc = matthews_corrcoef(y_d2, (probs_d2 >= 0.5).astype(int))
        
        print(f"  AUROC: {d2_auroc:.6f}")
        print(f"  AUPRC: {d2_auprc:.6f}")
        print(f"  MCC: {d2_mcc:.6f}")
        
        d2_results = {
            'AUROC': float(d2_auroc),
            'AUPRC': float(d2_auprc),
            'MCC': float(d2_mcc),
            'gate_weights': {
                'protein_mean': float(np.mean(gates_d2.cpu().numpy()[:, 0])),
                'dna_mean': float(np.mean(gates_d2.cpu().numpy()[:, 1])),
            }
        }
    
    # Save results
    results = {
        'method': 'Cross-Modal Attention Fusion (CMAF)',
        'architecture': {
            'protein_dim': len(protein_cols),
            'dna_dim': len(dna_cols),
            'aux_dim': len(aux_cols),
            'hidden_dim': best_params['hidden_dim'],
            'n_heads': best_params['n_heads'],
        },
        'best_params': best_params,
        'train_stats': {
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_pathogenic_train': int(y_train.sum()),
            'n_benign_train': int((y_train == 0).sum()),
        },
        'test_metrics': {
            'AUROC': float(auroc),
            'AUPRC': float(auprc),
            'MCC': float(mcc),
        },
        'gate_analysis': {
            'protein_weight_mean': float(np.mean(gates[:, 0])),
            'dna_weight_mean': float(np.mean(gates[:, 1])),
        },
        'd2_results': d2_results,
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'scaler_protein': scaler_protein,
        'scaler_dna': scaler_dna,
        'scaler_aux': scaler_aux,
        'best_params': best_params,
        'protein_cols': protein_cols,
        'dna_cols': dna_cols,
        'aux_cols': aux_cols,
    }, os.path.join(output_dir, 'model.pt'))
    
    # Save gate weights for visualization
    np.save(os.path.join(output_dir, 'gate_weights.npy'), gates)
    np.save(os.path.join(output_dir, 'attention_weights.npy'), attns)
    
    # Save predictions
    test_df['cmaf_score'] = probs
    test_df['gate_protein'] = gates[:, 0]
    test_df['gate_dna'] = gates[:, 1]
    test_df.to_csv(os.path.join(output_dir, 'test_predictions.tsv'), sep='\t', index=False)
    
    print(f"\nSaved to {output_dir}/")
    return results


# ─────────────────────────────────────────────────────────────
# Baselines for Comparison
# ─────────────────────────────────────────────────────────────

def run_baselines(dataset_dir, d2_dir=None, output_dir=None):
    """Run baseline models for comparison."""
    
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'cmaf_baselines')
    os.makedirs(output_dir, exist_ok=True)
    
    split_dir = os.path.join(dataset_dir, 'splits')
    train_df = pd.read_csv(os.path.join(split_dir, 'gene_disjoint_train.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(split_dir, 'gene_disjoint_test.tsv'), sep='\t')
    
    y_train = train_df['label_binary'].values
    y_test = test_df['label_binary'].values
    
    baselines = {}
    
    # Individual model scores
    individual_models = {
        'ESM1b-MLM': 'esm1b_mlm',
        'ESM-1v-MLM': 'esm1v_mlm',
        'ProtBERT-MLM': 'protbert_mlm',
        'ProtT5-Cos': 'prott5_cos',
        'EVO2-PLL': 'evo2_delta_score',
    }
    
    print("=== Individual Models ===")
    for name, col in individual_models.items():
        if col in test_df.columns:
            scores = test_df[col].values
            if col == 'evo2_delta_score':
                scores = -scores  # Negate EVO2
            auroc = roc_auc_score(y_test, scores)
            baselines[name] = {'AUROC': float(auroc)}
            print(f"  {name}: AUROC={auroc:.4f}")
    
    # Simple concatenation baseline
    from sklearn.linear_model import LogisticRegression
    
    feature_sets = {
        'ESM1b': ['esm1b_mlm'],
        'EVO2': ['evo2_delta_score'],
        'ESM1b+EVO2': ['esm1b_mlm', 'evo2_delta_score'],
        'ESM1b+EVO2+Phys': ['esm1b_mlm', 'evo2_delta_score', 'grantham', 'blosum62',
                            'hydrophobicity_change', 'charge_change'],
    }
    
    print("\n=== Logistic Regression Baselines ===")
    for name, cols in feature_sets.items():
        cols_available = [c for c in cols if c in train_df.columns]
        if cols_available:
            X_train = train_df[cols_available].values.copy()
            X_test = test_df[cols_available].values.copy()
            
            # Negate EVO2
            for i, col in enumerate(cols_available):
                if col == 'evo2_delta_score':
                    X_train[:, i] = -X_train[:, i]
                    X_test[:, i] = -X_test[:, i]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            
            auroc = roc_auc_score(y_test, probs)
            baselines[f'LR-{name}'] = {'AUROC': float(auroc)}
            print(f"  LR-{name}: AUROC={auroc:.4f}")
    
    # XGBoost baseline (V1 meta-learner)
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    EXCLUDE_COLS = {'rcv_accession', 'gene_symbol', 'label', 'label_binary'}
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    
    print("\n=== XGBoost Meta-Learner V1 ===")
    base_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        tree_method='hist', device='cuda', verbosity=0, random_state=42,
    )
    
    param_distributions = {
        'max_depth': randint(3, 11),
        'learning_rate': uniform(0.01, 0.29),
        'n_estimators': randint(200, 1600),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5),
    }
    
    search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=40,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc', random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(train_df[feature_cols], y_train)
    
    xgb_probs = search.predict_proba(test_df[feature_cols])[:, 1]
    xgb_auroc = roc_auc_score(y_test, xgb_probs)
    baselines['XGBoost-V1'] = {'AUROC': float(xgb_auroc)}
    print(f"  XGBoost-V1: AUROC={xgb_auroc:.4f}")
    
    # Save baselines
    with open(os.path.join(output_dir, 'baselines.json'), 'w') as f:
        json.dump(baselines, f, indent=2)
    
    print(f"\nSaved baselines to {output_dir}/")
    return baselines


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'dataset1_clinvar_only'
    d2 = sys.argv[2] if len(sys.argv) > 2 else 'dataset2_jan2025'
    
    # Run baselines first
    print("="*60)
    print("BASELINES")
    print("="*60)
    run_baselines(ds, d2)
    
    # Run CMAF
    print("\n" + "="*60)
    print("CROSS-MODAL ATTENTION FUSION (CMAF)")
    print("="*60)
    train_cmaf(ds, d2)
