"""
Phase 6B (Updated): Transferability with AUROC + Bootstrap CI
Tests predictive transferability with proper AUROC metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def bootstrap_auroc(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute AUROC with bootstrap confidence interval."""
    n = len(y_true)
    aurocs = []
    
    # Compute base AUROC
    base_auroc = roc_auc_score(y_true, y_pred_proba)
    
    # Bootstrap
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        
        # Skip if only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        auroc = roc_auc_score(y_true_boot, y_pred_boot)
        aurocs.append(auroc)
    
    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(aurocs, alpha * 100)
    ci_high = np.percentile(aurocs, (1 - alpha) * 100)
    
    return base_auroc, ci_low, ci_high


def run_transferability_with_auroc(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, 
                                   output_dir: Path):
    """Run Phase 6B with AUROC and bootstrap CI."""
    print("\n" + "="*70)
    print("PHASE 6B: TRANSFERABILITY WITH AUROC + BOOTSTRAP CI")
    print("="*70)
    
    # Prepare features (signal composition as proxy for phenotypic similarity)
    signal_cols = [c for c in mimic_df.columns if c.startswith('has_') and 
                   c not in ['has_hr_numeric', 'has_spo2_numeric', 'has_rr_numeric']]
    
    print(f"Using {len(signal_cols)} signal features")
    
    # Target: High-diversity monitoring (>=4 categories)
    def compute_diversity(row):
        return sum([row.get(col, False) for col in signal_cols])
    
    mimic_df = mimic_df.copy()
    uciq_df = uciq_df.copy()
    
    mimic_df['diversity'] = mimic_df.apply(compute_diversity, axis=1)
    uciq_df['diversity'] = uciq_df.apply(compute_diversity, axis=1)
    
    # Binary target: high diversity (>=4 signal categories)
    mimic_df['target'] = (mimic_df['diversity'] >= 4).astype(int)
    uciq_df['target'] = (uciq_df['diversity'] >= 4).astype(int)
    
    # Prepare data
    X_mimic = mimic_df[signal_cols].fillna(False).astype(int)
    y_mimic = mimic_df['target'].values
    
    X_uciq = uciq_df[signal_cols].fillna(False).astype(int)
    y_uciq = uciq_df['target'].values
    
    # Scale
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_uciq_scaled = scaler.transform(X_uciq)
    
    # Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = []
    
    print("\n" + "-"*70)
    print("TRANSFERABILITY RESULTS (AUROC with 95% CI)")
    print("-"*70)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # 1. Train on MIMIC, test on UCIQ
        model.fit(X_mimic_scaled, y_mimic)
        y_pred_proba_uciq = model.predict_proba(X_uciq_scaled)[:, 1]
        
        auroc_m2u, ci_low_m2u, ci_high_m2u = bootstrap_auroc(y_uciq, y_pred_proba_uciq)
        
        # 2. Train on UCIQ, test on MIMIC
        model.fit(X_uciq_scaled, y_uciq)
        y_pred_proba_mimic = model.predict_proba(X_mimic_scaled)[:, 1]
        
        auroc_u2m, ci_low_u2m, ci_high_u2m = bootstrap_auroc(y_mimic, y_pred_proba_mimic)
        
        # 3. Within-dataset AUROC (baselines)
        # MIMIC CV
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_aurocs_mimic = []
        for train_idx, val_idx in cv.split(X_mimic_scaled, y_mimic):
            X_train, X_val = X_mimic_scaled[train_idx], X_mimic_scaled[val_idx]
            y_train, y_val = y_mimic[train_idx], y_mimic[val_idx]
            
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            cv_aurocs_mimic.append(roc_auc_score(y_val, y_proba))
        
        cv_aurocs_uciq = []
        for train_idx, val_idx in cv.split(X_uciq_scaled, y_uciq):
            X_train, X_val = X_uciq_scaled[train_idx], X_uciq_scaled[val_idx]
            y_train, y_val = y_uciq[train_idx], y_uciq[val_idx]
            
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            cv_aurocs_uciq.append(roc_auc_score(y_val, y_proba))
        
        cv_mimic = np.mean(cv_aurocs_mimic)
        cv_uciq = np.mean(cv_aurocs_uciq)
        
        # Transfer gap
        gap_m2u = cv_mimic - auroc_m2u
        gap_u2m = cv_uciq - auroc_u2m
        
        print(f"  MIMIC → UCIQ: AUROC={auroc_m2u:.3f} [{ci_low_m2u:.3f}, {ci_high_m2u:.3f}]")
        print(f"  UCIQ → MIMIC: AUROC={auroc_u2m:.3f} [{ci_low_u2m:.3f}, {ci_high_u2m:.3f}]")
        print(f"  MIMIC CV (baseline): {cv_mimic:.3f}")
        print(f"  UCIQ CV (baseline): {cv_uciq:.3f}")
        print(f"  Transfer gap (M→U): {gap_m2u:.3f}")
        print(f"  Transfer gap (U→M): {gap_u2m:.3f}")
        
        results.append({
            'model': model_name,
            'mimic_to_uciq_auroc': auroc_m2u,
            'mimic_to_uciq_ci_low': ci_low_m2u,
            'mimic_to_uciq_ci_high': ci_high_m2u,
            'uciq_to_mimic_auroc': auroc_u2m,
            'uciq_to_mimic_ci_low': ci_low_u2m,
            'uciq_to_mimic_ci_high': ci_high_u2m,
            'mimic_cv_auroc': cv_mimic,
            'uciq_cv_auroc': cv_uciq,
            'gap_mimic_to_uciq': gap_m2u,
            'gap_uciq_to_mimic': gap_u2m
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'transferability_auroc_results.csv', index=False)
    
    # Visualization
    print("\nGenerating AUROC visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, label, color) in enumerate([
        ('mimic_cv_auroc', 'MIMIC CV', 'steelblue'),
        ('mimic_to_uciq_auroc', 'MIMIC→UCIQ', 'coral'),
        ('uciq_cv_auroc', 'UCIQ CV', 'seagreen')
    ]):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=label, color=color)
        
        # Add error bars for transfer
        if 'to' in metric:
            for j, (_, row) in enumerate(results_df.iterrows()):
                ci_low = row[metric.replace('auroc', 'ci_low')]
                ci_high = row[metric.replace('auroc', 'ci_high')]
                ax.errorbar(j + offset, row[metric], 
                          yerr=[[row[metric] - ci_low], [ci_high - row[metric]]],
                          fmt='none', color='black', capsize=5)
    
    ax.set_ylabel('AUROC')
    ax.set_title('Transferability: AUROC with 95% Bootstrap CI', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transferability_auroc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: transferability_auroc.png")
    
    # Summary
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)
    
    avg_gap_m2u = results_df['gap_mimic_to_uciq'].mean()
    avg_gap_u2m = results_df['gap_uciq_to_mimic'].mean()
    
    if avg_gap_m2u < 0.05 and avg_gap_u2m < 0.05:
        interpretation = "EXCELLENT transferability - minimal domain shift"
    elif avg_gap_m2u < 0.10 and avg_gap_u2m < 0.10:
        interpretation = "GOOD transferability - modest domain shift"
    elif avg_gap_m2u < 0.20 or avg_gap_u2m < 0.20:
        interpretation = "MODERATE transferability - significant domain shift"
    else:
        interpretation = "POOR transferability - substantial domain shift"
    
    print(f"Average transfer gap (MIMIC→UCIQ): {avg_gap_m2u:.3f}")
    print(f"Average transfer gap (UCIQ→MIMIC): {avg_gap_u2m:.3f}")
    print(f"Assessment: {interpretation}")
    
    print("\n" + "="*70)
    print("Phase 6B (AUROC) Complete")
    print("="*70)
    
    return {
        'transferability_results': results_df,
        'interpretation': interpretation
    }


if __name__ == "__main__":
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    results = run_transferability_with_auroc(mimic_df, uciq_df, output_dir)
