"""
Phase 7: Sensitivity Analysis
Tests robustness of findings to parameter variations
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


def bootstrap_prevalence(df: pd.DataFrame, signal_col: str, 
                         n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """Compute bootstrap confidence interval for signal prevalence."""
    n = len(df)
    prevalences = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = df.sample(n=n, replace=True)
        prev = sample[signal_col].mean()
        prevalences.append(prev)
    
    prevalences = np.array(prevalences)
    
    alpha = (1 - ci) / 2
    ci_low = np.percentile(prevalences, alpha * 100)
    ci_high = np.percentile(prevalences, (1 - alpha) * 100)
    
    return {
        'mean': prevalences.mean(),
        'std': prevalences.std(),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'bootstrap_se': prevalences.std() / np.sqrt(n_bootstrap)
    }


def run_sensitivity_analysis(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, 
                             output_dir: Path):
    """Run Phase 7: Sensitivity analysis with bootstrap confidence intervals."""
    print("\n" + "="*70)
    print("PHASE 7: SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Key signals for sensitivity analysis
    key_signals = [
        'has_ecg', 'has_ppg', 'has_resp', 'has_abp_invasive', 
        'has_nibp', 'has_co2', 'has_icp', 'has_bis_eeg'
    ]
    
    # Filter to existing columns
    mimic_signals = [s for s in key_signals if s in mimic_df.columns]
    uciq_signals = [s for s in key_signals if s in uciq_df.columns]
    common_signals = list(set(mimic_signals) & set(uciq_signals))
    
    print(f"\nComputing bootstrap confidence intervals for {len(common_signals)} signals...")
    print(f"Bootstrap samples: 1000, Confidence level: 95%")
    
    results = []
    
    for signal in common_signals:
        # MIMIC bootstrap
        mimic_boot = bootstrap_prevalence(mimic_df, signal, n_bootstrap=1000)
        
        # UCIQ bootstrap
        uciq_boot = bootstrap_prevalence(uciq_df, signal, n_bootstrap=1000)
        
        # Check if CIs overlap
        overlap = not (mimic_boot['ci_high'] < uciq_boot['ci_low'] or 
                       uciq_boot['ci_high'] < mimic_boot['ci_low'])
        
        results.append({
            'signal': signal.replace('has_', ''),
            'mimic_prevalence': mimic_df[signal].mean(),
            'mimic_ci_low': mimic_boot['ci_low'],
            'mimic_ci_high': mimic_boot['ci_high'],
            'mimic_se': mimic_boot['bootstrap_se'],
            'uciq_prevalence': uciq_df[signal].mean(),
            'uciq_ci_low': uciq_boot['ci_low'],
            'uciq_ci_high': uciq_boot['ci_high'],
            'uciq_se': uciq_boot['bootstrap_se'],
            'ci_overlap': overlap,
            'significant_diff': not overlap
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)
    
    print("\n" + "-"*70)
    print("SENSITIVITY RESULTS (Bootstrap 95% CI)")
    print("-"*70)
    
    for _, row in results_df.iterrows():
        sig = row['signal']
        m_prev = row['mimic_prevalence'] * 100
        m_ci = f"[{row['mimic_ci_low']*100:.1f}%, {row['mimic_ci_high']*100:.1f}%]"
        u_prev = row['uciq_prevalence'] * 100
        u_ci = f"[{row['uciq_ci_low']*100:.1f}%, {row['uciq_ci_high']*100:.1f}%]"
        diff = "✓" if row['significant_diff'] else "✗"
        
        print(f"\n{sig}:")
        print(f"  MIMIC: {m_prev:.1f}% {m_ci}")
        print(f"  UCIQ:  {u_prev:.1f}% {u_ci}")
        print(f"  Significant difference: {diff}")
    
    # Visualization
    print("\nGenerating sensitivity plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(results_df))
    
    # Plot CIs as error bars
    for i, (_, row) in enumerate(results_df.iterrows()):
        # MIMIC
        ax.errorbar(row['mimic_prevalence'] * 100, i - 0.15,
                   xerr=[[(row['mimic_prevalence'] - row['mimic_ci_low']) * 100],
                         [(row['mimic_ci_high'] - row['mimic_prevalence']) * 100]],
                   fmt='o', color='steelblue', capsize=5, capthick=2,
                   elinewidth=2, markersize=8, label='MIMIC' if i == 0 else '')
        
        # UCIQ
        ax.errorbar(row['uciq_prevalence'] * 100, i + 0.15,
                   xerr=[[(row['uciq_prevalence'] - row['uciq_ci_low']) * 100],
                         [(row['uciq_ci_high'] - row['uciq_prevalence']) * 100]],
                   fmt='s', color='coral', capsize=5, capthick=2,
                   elinewidth=2, markersize=8, label='UCIQ' if i == 0 else '')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace('has_', '') for s in results_df['signal']])
    ax.set_xlabel('Prevalence (%)')
    ax.set_title('Signal Prevalence with 95% Bootstrap CI', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.axvline(50, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_prevalence_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sensitivity_prevalence_ci.png")
    
    # Summary
    n_significant = results_df['significant_diff'].sum()
    n_total = len(results_df)
    
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"Signals tested: {n_total}")
    print(f"Significant differences: {n_significant} ({n_significant/n_total*100:.1f}%)")
    print(f"Robust findings (overlapping CIs): {n_total - n_significant}")
    
    if n_significant / n_total > 0.5:
        print("\nConclusion: Most signal prevalence differences are ROBUST to sampling variation.")
    else:
        print("\nConclusion: Many differences may be due to sampling variation.")
    
    print("\n" + "="*70)
    print("Phase 7 Complete")
    print("="*70)
    
    return {
        'sensitivity_results': results_df,
        'n_significant': n_significant,
        'n_total': n_total
    }


if __name__ == "__main__":
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    results = run_sensitivity_analysis(mimic_df, uciq_df, output_dir)
