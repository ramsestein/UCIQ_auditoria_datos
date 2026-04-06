"""
Phase 2.4: Diversity Metrics Analysis
Computes Shannon entropy and signal diversity per record
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns


def shannon_entropy(presence_vector: np.ndarray) -> float:
    """Compute Shannon entropy of signal presence distribution"""
    # Count present vs absent
    present = np.sum(presence_vector)
    absent = len(presence_vector) - present
    
    if present == 0 or absent == 0:
        return 0.0
    
    total = len(presence_vector)
    p_present = present / total
    p_absent = absent / total
    
    # Shannon entropy: -sum(p * log(p))
    entropy = -(p_present * np.log2(p_present) + p_absent * np.log2(p_absent))
    return entropy


def compute_diversity_metrics(df: pd.DataFrame, signal_cols: list) -> pd.DataFrame:
    """Compute diversity metrics for each record"""
    results = []
    
    for idx, row in df.iterrows():
        presence = np.array([row.get(col, False) for col in signal_cols])
        
        n_categories = len(signal_cols)
        n_present = np.sum(presence)
        
        # Diversity ratio
        diversity_ratio = n_present / n_categories if n_categories > 0 else 0
        
        # Shannon entropy
        entropy = shannon_entropy(presence)
        
        results.append({
            'record_id': row.get('record_id', idx),
            'dataset': row.get('dataset', 'unknown'),
            'n_signal_categories': n_present,
            'diversity_ratio': diversity_ratio,
            'shannon_entropy': entropy
        })
    
    return pd.DataFrame(results)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size"""
    x = np.array(x)
    y = np.array(y)
    
    # All pairwise comparisons
    n_x = len(x)
    n_y = len(y)
    
    if n_x == 0 or n_y == 0:
        return 0.0
    
    # Vectorized computation
    x_expanded = x[:, np.newaxis]
    y_expanded = y[np.newaxis, :]
    
    greater = np.sum(x_expanded > y_expanded)
    less = np.sum(x_expanded < y_expanded)
    
    delta = (greater - less) / (n_x * n_y)
    return delta


def run_diversity_analysis(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, output_dir: Path):
    """Run complete diversity analysis (Phase 2.4)"""
    print("\n" + "="*70)
    print("PHASE 2.4: DIVERSITY METRICS ANALYSIS")
    print("="*70)
    
    # Define signal categories
    signal_categories = [
        'has_ecg', 'has_ppg', 'has_resp', 'has_abp_invasive', 'has_nibp',
        'has_co2', 'has_icp', 'has_bis_eeg', 'has_temperature', 
        'has_ventilation', 'has_cvp', 'has_pap'
    ]
    
    # Filter to existing columns
    common_cols = [c for c in signal_categories if c in mimic_df.columns and c in uciq_df.columns]
    print(f"\nAnalyzing diversity across {len(common_cols)} signal categories")
    
    # Compute diversity metrics
    print("\nComputing diversity metrics per record...")
    mimic_div = compute_diversity_metrics(mimic_df, common_cols)
    uciq_div = compute_diversity_metrics(uciq_df, common_cols)
    
    # Add to original dataframes
    mimic_df = mimic_df.copy()
    uciq_df = uciq_df.copy()
    mimic_df['diversity_ratio'] = mimic_div['diversity_ratio'].values
    mimic_df['shannon_entropy'] = mimic_div['shannon_entropy'].values
    uciq_df['diversity_ratio'] = uciq_div['diversity_ratio'].values
    uciq_df['shannon_entropy'] = uciq_div['shannon_entropy'].values
    
    # Summary statistics
    print("\n" + "-"*70)
    print("DIVERSITY SUMMARY")
    print("-"*70)
    
    summary_stats = pd.DataFrame({
        'MIMIC': {
            'n_records': len(mimic_div),
            'median_categories': mimic_div['n_signal_categories'].median(),
            'mean_categories': mimic_div['n_signal_categories'].mean(),
            'median_diversity': mimic_div['diversity_ratio'].median(),
            'mean_diversity': mimic_div['diversity_ratio'].mean(),
            'median_entropy': mimic_div['shannon_entropy'].median(),
        },
        'UCIQ': {
            'n_records': len(uciq_div),
            'median_categories': uciq_div['n_signal_categories'].median(),
            'mean_categories': uciq_div['n_signal_categories'].mean(),
            'median_diversity': uciq_div['diversity_ratio'].median(),
            'mean_diversity': uciq_div['diversity_ratio'].mean(),
            'median_entropy': uciq_div['shannon_entropy'].median(),
        }
    })
    
    print(summary_stats.to_string())
    
    # Statistical tests
    print("\n" + "-"*70)
    print("STATISTICAL COMPARISONS")
    print("-"*70)
    
    # Mann-Whitney U test for n_signal_categories
    mw_stat, mw_p = mannwhitneyu(
        mimic_div['n_signal_categories'], 
        uciq_div['n_signal_categories'],
        alternative='two-sided'
    )
    
    # KS test for diversity ratio
    ks_stat, ks_p = ks_2samp(
        mimic_div['diversity_ratio'], 
        uciq_div['diversity_ratio']
    )
    
    # Cliff's delta
    cd = cliffs_delta(
        mimic_div['n_signal_categories'].values,
        uciq_div['n_signal_categories'].values
    )
    
    # Interpret Cliff's delta
    if abs(cd) < 0.147:
        cd_interpretation = "negligible"
    elif abs(cd) < 0.33:
        cd_interpretation = "small"
    elif abs(cd) < 0.474:
        cd_interpretation = "medium"
    else:
        cd_interpretation = "large"
    
    print(f"\nMann-Whitney U test (signal categories):")
    print(f"  Statistic: {mw_stat:.2f}, p-value: {mw_p:.2e}")
    
    print(f"\nKolmogorov-Smirnov test (diversity ratio):")
    print(f"  Statistic: {ks_stat:.3f}, p-value: {ks_p:.2e}")
    
    print(f"\nCliff's delta (effect size):")
    print(f"  Delta: {cd:.3f} ({cd_interpretation} effect)")
    print(f"  Positive = MIMIC higher, Negative = UCIQ higher")
    
    # Save results
    summary_stats.to_csv(output_dir / 'diversity_summary.csv')
    
    # Create violin plots
    print("\nGenerating violin plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for plotting
    plot_data = pd.concat([
        mimic_div.assign(dataset='MIMIC'),
        uciq_div.assign(dataset='UCIQ')
    ])
    
    # Violin plot 1: Number of signal categories
    sns.violinplot(data=plot_data, x='dataset', y='n_signal_categories', ax=axes[0])
    axes[0].set_title('Distribution of Signal Categories per Record', fontweight='bold')
    axes[0].set_ylabel('Number of Signal Categories')
    axes[0].set_xlabel('Dataset')
    
    # Add median lines
    for i, dataset in enumerate(['MIMIC', 'UCIQ']):
        median = plot_data[plot_data['dataset'] == dataset]['n_signal_categories'].median()
        axes[0].hlines(median, i-0.2, i+0.2, colors='red', linestyles='--', linewidth=2)
    
    # Violin plot 2: Diversity ratio
    sns.violinplot(data=plot_data, x='dataset', y='diversity_ratio', ax=axes[1])
    axes[1].set_title('Distribution of Diversity Ratio', fontweight='bold')
    axes[1].set_ylabel('Diversity Ratio (present/total)')
    axes[1].set_xlabel('Dataset')
    
    # Add median lines
    for i, dataset in enumerate(['MIMIC', 'UCIQ']):
        median = plot_data[plot_data['dataset'] == dataset]['diversity_ratio'].median()
        axes[1].hlines(median, i-0.2, i+0.2, colors='red', linestyles='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diversity_violin_plots.png")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'metric': ['n_signal_categories', 'diversity_ratio', 'shannon_entropy'],
        'mimic_median': [
            mimic_div['n_signal_categories'].median(),
            mimic_div['diversity_ratio'].median(),
            mimic_div['shannon_entropy'].median()
        ],
        'uciq_median': [
            uciq_div['n_signal_categories'].median(),
            uciq_div['diversity_ratio'].median(),
            uciq_div['shannon_entropy'].median()
        ],
        'mannwhitney_stat': [mw_stat, np.nan, np.nan],
        'mannwhitney_p': [mw_p, np.nan, np.nan],
        'ks_stat': [np.nan, ks_stat, np.nan],
        'ks_p': [np.nan, ks_p, np.nan],
        'cliffs_delta': [cd, np.nan, np.nan],
        'effect_size': [cd_interpretation, '', '']
    })
    results_df.to_csv(output_dir / 'diversity_comparison.csv', index=False)
    
    print("\n" + "="*70)
    print("Phase 2.4 Complete")
    print("="*70)
    
    return {
        'summary': summary_stats,
        'mimic_diversity': mimic_div,
        'uciq_diversity': uciq_div,
        'statistical_tests': {
            'mannwhitney': (mw_stat, mw_p),
            'ks_test': (ks_stat, ks_p),
            'cliffs_delta': cd
        }
    }


if __name__ == "__main__":
    # Load processed data
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    # Run analysis
    results = run_diversity_analysis(mimic_df, uciq_df, output_dir)
