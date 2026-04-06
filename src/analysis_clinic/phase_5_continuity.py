"""
Phase 5: Continuity and Fragmentation Analysis
Analyzes recording duration gaps, segment continuity, and fragmentation patterns
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def analyze_continuity(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Analyze continuity metrics for a dataset."""
    
    # Duration statistics
    durations = df['duration_hours'].dropna()
    
    if len(durations) == 0:
        return {
            'dataset': dataset_name,
            'n_records': len(df),
            'median_duration_hours': 0,
            'mean_duration_hours': 0,
            'std_duration_hours': 0,
            'p25_duration': 0,
            'p75_duration': 0,
            'fragmentation_index': 0
        }
    
    return {
        'dataset': dataset_name,
        'n_records': len(df),
        'median_duration_hours': durations.median(),
        'mean_duration_hours': durations.mean(),
        'std_duration_hours': durations.std(),
        'p25_duration': durations.quantile(0.25),
        'p75_duration': durations.quantile(0.75),
        'fragmentation_index': durations.std() / durations.mean() if durations.mean() > 0 else 0
    }


def run_continuity_analysis(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, output_dir: Path):
    """Run Phase 5: Continuity and fragmentation analysis."""
    print("\n" + "="*70)
    print("PHASE 5: CONTINUITY AND FRAGMENTATION ANALYSIS")
    print("="*70)
    
    # Analyze continuity for each dataset
    print("\nAnalyzing recording continuity...")
    mimic_continuity = analyze_continuity(mimic_df, 'MIMIC')
    uciq_continuity = analyze_continuity(uciq_df, 'UCIQ')
    
    # Create summary
    summary = pd.DataFrame([mimic_continuity, uciq_continuity]).set_index('dataset')
    
    print("\n" + "-"*70)
    print("CONTINUITY SUMMARY")
    print("-"*70)
    print(summary.to_string())
    
    # Save summary
    summary.to_csv(output_dir / 'continuity_summary.csv')
    
    # Statistical comparison
    mimic_durations = mimic_df['duration_hours'].dropna()
    uciq_durations = uciq_df['duration_hours'].dropna()
    
    if len(mimic_durations) > 0 and len(uciq_durations) > 0:
        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(
            mimic_durations, uciq_durations, alternative='two-sided'
        )
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(mimic_durations, uciq_durations)
        
        print("\n" + "-"*70)
        print("STATISTICAL COMPARISON (Duration)")
        print("-"*70)
        print(f"Mann-Whitney U: statistic={mw_stat:.2f}, p-value={mw_p:.2e}")
        print(f"Kolmogorov-Smirnov: statistic={ks_stat:.3f}, p-value={ks_p:.2e}")
        
        # Save comparison
        comparison = pd.DataFrame({
            'test': ['Mann-Whitney U', 'Kolmogorov-Smirnov'],
            'statistic': [mw_stat, ks_stat],
            'p_value': [mw_p, ks_p]
        })
        comparison.to_csv(output_dir / 'continuity_comparison.csv', index=False)
    
    # Generate duration distribution plot
    print("\nGenerating duration distribution plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram comparison
    bins = np.linspace(0, min(50, max(
        mimic_durations.max() if len(mimic_durations) > 0 else 50,
        uciq_durations.max() if len(uciq_durations) > 0 else 50
    )), 50)
    
    axes[0].hist(mimic_durations, bins=bins, alpha=0.6, label='MIMIC', color='steelblue', density=True)
    axes[0].hist(uciq_durations, bins=bins, alpha=0.6, label='UCIQ', color='coral', density=True)
    axes[0].set_xlabel('Duration (hours)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Recording Duration Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].axvline(mimic_durations.median() if len(mimic_durations) > 0 else 0, 
                   color='steelblue', linestyle='--', label=f'MIMIC median: {mimic_durations.median():.1f}h')
    axes[0].axvline(uciq_durations.median() if len(uciq_durations) > 0 else 0, 
                   color='coral', linestyle='--', label=f'UCIQ median: {uciq_durations.median():.1f}h')
    
    # Box plot
    plot_data = pd.concat([
        pd.DataFrame({'duration_hours': mimic_durations, 'dataset': 'MIMIC'}),
        pd.DataFrame({'duration_hours': uciq_durations, 'dataset': 'UCIQ'})
    ])
    
    sns.boxplot(data=plot_data, x='dataset', y='duration_hours', ax=axes[1])
    axes[1].set_title('Duration Distribution (Box Plot)', fontweight='bold')
    axes[1].set_ylabel('Duration (hours)')
    axes[1].set_xlabel('Dataset')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: duration_distribution.png")
    
    # Duration categories analysis
    print("\n" + "-"*70)
    print("DURATION CATEGORIES")
    print("-"*70)
    
    def categorize_duration(durations):
        categories = pd.cut(durations, 
                          bins=[0, 1, 6, 24, 72, float('inf')],
                          labels=['<1h', '1-6h', '6-24h', '24-72h', '>72h'])
        return categories.value_counts()
    
    mimic_cats = categorize_duration(mimic_durations)
    uciq_cats = categorize_duration(uciq_durations)
    
    duration_cats = pd.DataFrame({
        'MIMIC_n': mimic_cats,
        'MIMIC_pct': mimic_cats / len(mimic_durations) * 100 if len(mimic_durations) > 0 else 0,
        'UCIQ_n': uciq_cats,
        'UCIQ_pct': uciq_cats / len(uciq_durations) * 100 if len(uciq_durations) > 0 else 0
    }).fillna(0)
    
    print(duration_cats.to_string())
    duration_cats.to_csv(output_dir / 'duration_categories.csv')
    
    print("\n" + "="*70)
    print("Phase 5 Complete")
    print("="*70)
    
    return {
        'continuity_summary': summary,
        'duration_categories': duration_cats
    }


if __name__ == "__main__":
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    results = run_continuity_analysis(mimic_df, uciq_df, output_dir)
