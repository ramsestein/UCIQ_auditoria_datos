"""
Phase 6B: Transferability Experiment
Tests model transferability between MIMIC and UCIQ datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def run_transferability_experiment(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, 
                                   output_dir: Path):
    """Run Phase 6B: Transferability experiment.
    
    Train on one dataset, test on the other for phenotype prediction.
    """
    print("\n" + "="*70)
    print("PHASE 6B: TRANSFERABILITY EXPERIMENT")
    print("="*70)
    
    # Prepare features
    signal_cols = [c for c in mimic_df.columns if c.startswith('has_')]
    
    # Add target: high-diversity phenotype (4+ signal categories)
    def compute_diversity(row):
        return sum([row.get(col, False) for col in signal_cols])
    
    mimic_df = mimic_df.copy()
    uciq_df = uciq_df.copy()
    
    mimic_df['diversity'] = mimic_df.apply(compute_diversity, axis=1)
    uciq_df['diversity'] = uciq_df.apply(compute_diversity, axis=1)
    
    # Binary target: high diversity (>=4) vs low diversity (<4)
    mimic_df['high_diversity'] = (mimic_df['diversity'] >= 4).astype(int)
    uciq_df['high_diversity'] = (uciq_df['diversity'] >= 4).astype(int)
    
    # Prepare data
    X_mimic = mimic_df[signal_cols].fillna(False).astype(int)
    y_mimic = mimic_df['high_diversity'].values
    
    X_uciq = uciq_df[signal_cols].fillna(False).astype(int)
    y_uciq = uciq_df['high_diversity'].values
    
    # Scale features
    scaler = StandardScaler()
    X_mimic_scaled = scaler.fit_transform(X_mimic)
    X_uciq_scaled = scaler.transform(X_uciq)
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = []
    
    print("\n" + "-"*70)
    print("TRANSFERABILITY RESULTS")
    print("-"*70)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # 1. Train on MIMIC, test on UCIQ
        model.fit(X_mimic_scaled, y_mimic)
        y_pred_uciq = model.predict(X_uciq_scaled)
        
        acc_mimic_to_uciq = accuracy_score(y_uciq, y_pred_uciq)
        f1_mimic_to_uciq = f1_score(y_uciq, y_pred_uciq, zero_division=0)
        
        # 2. Train on UCIQ, test on MIMIC
        model.fit(X_uciq_scaled, y_uciq)
        y_pred_mimic = model.predict(X_mimic_scaled)
        
        acc_uciq_to_mimic = accuracy_score(y_mimic, y_pred_mimic)
        f1_uciq_to_mimic = f1_score(y_mimic, y_pred_mimic, zero_division=0)
        
        # 3. Within-dataset performance (baselines)
        # MIMIC CV
        from sklearn.model_selection import cross_val_score
        cv_mimic = cross_val_score(model, X_mimic_scaled, y_mimic, cv=5).mean()
        
        # UCIQ CV
        cv_uciq = cross_val_score(model, X_uciq_scaled, y_uciq, cv=5).mean()
        
        print(f"  MIMIC → UCIQ: Accuracy={acc_mimic_to_uciq:.3f}, F1={f1_mimic_to_uciq:.3f}")
        print(f"  UCIQ → MIMIC: Accuracy={acc_uciq_to_mimic:.3f}, F1={f1_uciq_to_mimic:.3f}")
        print(f"  MIMIC CV (baseline): {cv_mimic:.3f}")
        print(f"  UCIQ CV (baseline): {cv_uciq:.3f}")
        
        # Transfer gap
        gap_mimic_to_uciq = cv_mimic - acc_mimic_to_uciq
        gap_uciq_to_mimic = cv_uciq - acc_uciq_to_mimic
        
        print(f"  Transfer gap (MIMIC→UCIQ): {gap_mimic_to_uciq:.3f}")
        print(f"  Transfer gap (UCIQ→MIMIC): {gap_uciq_to_mimic:.3f}")
        
        results.append({
            'model': model_name,
            'mimic_to_uciq_acc': acc_mimic_to_uciq,
            'mimic_to_uciq_f1': f1_mimic_to_uciq,
            'uciq_to_mimic_acc': acc_uciq_to_mimic,
            'uciq_to_mimic_f1': f1_uciq_to_mimic,
            'mimic_cv': cv_mimic,
            'uciq_cv': cv_uciq,
            'gap_mimic_to_uciq': gap_mimic_to_uciq,
            'gap_uciq_to_mimic': gap_uciq_to_mimic
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'transferability_results.csv', index=False)
    
    # Visualization
    print("\nGenerating transferability visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Transfer accuracy comparison
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(['mimic_cv', 'mimic_to_uciq_acc', 'uciq_cv']):
        offset = (i - 1) * width
        axes[0].bar(x + offset, results_df[metric], width, 
                   label=['MIMIC CV', 'MIMIC→UCIQ', 'UCIQ CV'][i])
    
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Transfer Accuracy: MIMIC → UCIQ', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df['model'])
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Reverse transfer
    for i, metric in enumerate(['uciq_cv', 'uciq_to_mimic_acc', 'mimic_cv']):
        offset = (i - 1) * width
        axes[1].bar(x + offset, results_df[metric], width,
                   label=['UCIQ CV', 'UCIQ→MIMIC', 'MIMIC CV'][i])
    
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Transfer Accuracy: UCIQ → MIMIC', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['model'])
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transferability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: transferability_comparison.png")
    
    # Summary interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)
    
    avg_gap_mimic_to_uciq = results_df['gap_mimic_to_uciq'].mean()
    avg_gap_uciq_to_mimic = results_df['gap_uciq_to_mimic'].mean()
    
    if avg_gap_mimic_to_uciq < 0.1 and avg_gap_uciq_to_mimic < 0.1:
        interpretation = "EXCELLENT transferability - models generalize well"
    elif avg_gap_mimic_to_uciq < 0.2 and avg_gap_uciq_to_mimic < 0.2:
        interpretation = "GOOD transferability - minor domain shift"
    elif avg_gap_mimic_to_uciq < 0.3 or avg_gap_uciq_to_mimic < 0.3:
        interpretation = "MODERATE transferability - significant domain shift"
    else:
        interpretation = "POOR transferability - substantial domain shift"
    
    print(f"Average transfer gap (MIMIC→UCIQ): {avg_gap_mimic_to_uciq:.3f}")
    print(f"Average transfer gap (UCIQ→MIMIC): {avg_gap_uciq_to_mimic:.3f}")
    print(f"Assessment: {interpretation}")
    
    print("\n" + "="*70)
    print("Phase 6B Complete")
    print("="*70)
    
    return {
        'transferability_results': results_df,
        'interpretation': interpretation
    }


if __name__ == "__main__":
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    results = run_transferability_experiment(mimic_df, uciq_df, output_dir)
