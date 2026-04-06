"""
Phase 2.3: Signal Co-occurrence Analysis
Computes co-occurrence matrices and Jaccard distances between MIMIC and UCIQ
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


def compute_co_occurrence(df: pd.DataFrame, signal_cols: list) -> pd.DataFrame:
    """Compute co-occurrence matrix for signal pairs"""
    n = len(signal_cols)
    cooccur = np.zeros((n, n))
    
    for i, sig_i in enumerate(signal_cols):
        for j, sig_j in enumerate(signal_cols):
            if i <= j:  # Compute only upper triangle
                # Both signals present
                both_present = ((df[sig_i] == True) & (df[sig_j] == True)).sum()
                cooccur[i, j] = both_present / len(df)
                cooccur[j, i] = cooccur[i, j]  # Symmetric
    
    return pd.DataFrame(cooccur, index=signal_cols, columns=signal_cols)


def jaccard_distance(mimic_cooccur: pd.DataFrame, uciq_cooccur: pd.DataFrame) -> float:
    """Compute mean Jaccard distance between co-occurrence profiles"""
    # Flatten matrices (upper triangle only, excluding diagonal)
    n = mimic_cooccur.shape[0]
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    
    mimic_flat = mimic_cooccur.values[mask]
    uciq_flat = uciq_cooccur.values[mask]
    
    # Jaccard similarity for each pair: intersection / union
    # Here we use the co-occurrence values directly
    jaccard_sims = []
    for m, u in zip(mimic_flat, uciq_flat):
        if m + u > 0:
            jaccard_sims.append(min(m, u) / max(m, u))
        else:
            jaccard_sims.append(1.0)  # Both 0 = identical
    
    return 1 - np.mean(jaccard_sims)


def plot_co_occurrence_heatmaps(mimic_cooccur: pd.DataFrame, uciq_cooccur: pd.DataFrame, 
                                 output_dir: Path):
    """Generate side-by-side heatmaps of co-occurrence"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Clean column names for display
    clean_cols = [c.replace('has_', '').replace('_', ' ').upper() for c in mimic_cooccur.columns]
    
    # MIMIC heatmap
    sns.heatmap(mimic_cooccur.values, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=clean_cols, yticklabels=clean_cols,
                vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Co-occurrence'})
    axes[0].set_title('MIMIC (US MICU)\nCo-occurrence Matrix', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)
    
    # UCIQ heatmap
    sns.heatmap(uciq_cooccur.values, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=clean_cols, yticklabels=clean_cols,
                vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Co-occurrence'})
    axes[1].set_title('UCIQ (Barcelona SICU)\nCo-occurrence Matrix', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'co_occurrence_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_co_occurrence_analysis(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, output_dir: Path):
    """Run complete co-occurrence analysis (Phase 2.3)"""
    print("\n" + "="*70)
    print("PHASE 2.3: SIGNAL CO-OCCURRENCE ANALYSIS")
    print("="*70)
    
    # Define signal categories for analysis (12 categories)
    signal_categories = [
        'has_ecg', 'has_ppg', 'has_resp', 'has_abp_invasive', 'has_nibp',
        'has_co2', 'has_icp', 'has_bis_eeg', 'has_temperature', 
        'has_ventilation', 'has_cvp', 'has_pap'
    ]
    
    # Filter to existing columns
    mimic_cols = [c for c in signal_categories if c in mimic_df.columns]
    uciq_cols = [c for c in signal_categories if c in uciq_df.columns]
    common_cols = list(set(mimic_cols) & set(uciq_cols))
    
    print(f"\nAnalyzing {len(common_cols)} signal categories:")
    for col in common_cols:
        print(f"  - {col.replace('has_', '')}")
    
    # Compute co-occurrence matrices
    print("\nComputing co-occurrence matrices...")
    mimic_cooccur = compute_co_occurrence(mimic_df, common_cols)
    uciq_cooccur = compute_co_occurrence(uciq_df, common_cols)
    
    # Save matrices
    mimic_cooccur.to_csv(output_dir / 'co_occurrence_mimic.csv')
    uciq_cooccur.to_csv(output_dir / 'co_occurrence_uciq.csv')
    print(f"  Saved: co_occurrence_mimic.csv")
    print(f"  Saved: co_occurrence_uciq.csv")
    
    # Compute Jaccard distance
    jaccard_dist = jaccard_distance(mimic_cooccur, uciq_cooccur)
    print(f"\nMean Jaccard Distance: {jaccard_dist:.3f}")
    print(f"  (0 = identical profiles, 1 = completely different)")
    
    # Generate heatmaps
    print("\nGenerating heatmaps...")
    plot_co_occurrence_heatmaps(mimic_cooccur, uciq_cooccur, output_dir)
    print(f"  Saved: co_occurrence_heatmaps.png")
    
    # Summary statistics
    print("\n" + "-"*70)
    print("TOP CO-OCCURRENCE PAIRS:")
    print("-"*70)
    
    # Find highest co-occurrence pairs for each dataset
    def get_top_pairs(cooccur_df, dataset_name, n=5):
        pairs = []
        cols = cooccur_df.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append({
                    'pair': f"{cols[i].replace('has_', '')} + {cols[j].replace('has_', '')}",
                    'co_occurrence': cooccur_df.iloc[i, j]
                })
        pairs_df = pd.DataFrame(pairs).sort_values('co_occurrence', ascending=False)
        return pairs_df.head(n)
    
    mimic_top = get_top_pairs(mimic_cooccur, 'MIMIC')
    uciq_top = get_top_pairs(uciq_cooccur, 'UCIQ')
    
    print("\nMIMIC (US MICU):")
    for _, row in mimic_top.iterrows():
        print(f"  {row['pair']}: {row['co_occurrence']:.1%}")
    
    print("\nUCIQ (Barcelona SICU):")
    for _, row in uciq_top.iterrows():
        print(f"  {row['pair']}: {row['co_occurrence']:.1%}")
    
    print("\n" + "="*70)
    print("Phase 2.3 Complete")
    print("="*70)
    
    return {
        'mimic_cooccurrence': mimic_cooccur,
        'uciq_cooccurrence': uciq_cooccur,
        'jaccard_distance': jaccard_dist,
        'top_pairs_mimic': mimic_top,
        'top_pairs_uciq': uciq_top
    }


if __name__ == "__main__":
    # Load processed data
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    # Run analysis
    results = run_co_occurrence_analysis(mimic_df, uciq_df, output_dir)
