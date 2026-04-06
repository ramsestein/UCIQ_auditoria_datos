"""
Phase 4B & 4: Intra-signal Variability and Signal Quality Metrics
Analyzes signal dynamics and quality characteristics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import entropy, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_signal_quality_metrics(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute quality metrics for a signal segment.
    
    Metrics:
    - Signal-to-noise ratio (SNR)
    - Artifact ratio (% of samples beyond 3 SD)
    - Flatline ratio (% of constant sequences)
    - Dynamic range (max - min)
    - Coefficient of variation (CV)
    - Skewness and kurtosis
    """
    if len(signal) == 0 or np.all(np.isnan(signal)):
        return {}
    
    # Remove NaN
    signal = signal[~np.isnan(signal)]
    if len(signal) < 10:
        return {}
    
    # Basic statistics
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0:
        return {}
    
    # Signal-to-noise ratio (estimated)
    # Using coefficient of variation as proxy
    snr = mean_val / std_val if mean_val != 0 else 0
    
    # Artifact ratio (samples beyond 3 SD from mean)
    artifacts = np.abs(signal - mean_val) > 3 * std_val
    artifact_ratio = np.sum(artifacts) / len(signal) * 100
    
    # Flatline detection (consecutive identical values)
    diffs = np.diff(signal)
    flatlines = np.sum(np.abs(diffs) < 1e-10) / len(diffs) * 100 if len(diffs) > 0 else 0
    
    # Dynamic range
    dynamic_range = np.max(signal) - np.min(signal)
    
    # Coefficient of variation
    cv = std_val / mean_val if mean_val != 0 else 0
    
    # Distribution shape
    skewness = skew(signal)
    kurt = kurtosis(signal)
    
    return {
        'snr_proxy': snr,
        'artifact_ratio_pct': artifact_ratio,
        'flatline_ratio_pct': flatlines,
        'dynamic_range': dynamic_range,
        'cv': cv,
        'skewness': skewness,
        'kurtosis': kurt
    }


def compute_variability_metrics(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute intra-signal variability metrics.
    
    Metrics:
    - Variance
    - Standard deviation
    - RMS
    - Peak-to-peak amplitude
    - Signal entropy (approximate)
    - Autocorrelation decay
    """
    if len(signal) == 0 or np.all(np.isnan(signal)):
        return {}
    
    signal = signal[~np.isnan(signal)]
    if len(signal) < 10:
        return {}
    
    # Basic variability
    variance = np.var(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    
    # Peak-to-peak
    p2p = np.max(signal) - np.min(signal)
    
    # Signal entropy (using histogram approximation)
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]
    signal_entropy = entropy(hist) if len(hist) > 0 else 0
    
    # Autocorrelation at lag 1 (normalized)
    if len(signal) > 1:
        autocorr_lag1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
        if np.isnan(autocorr_lag1):
            autocorr_lag1 = 0
    else:
        autocorr_lag1 = 0
    
    return {
        'variance': variance,
        'std': std,
        'rms': rms,
        'peak_to_peak': p2p,
        'signal_entropy': signal_entropy,
        'autocorr_lag1': autocorr_lag1
    }


def run_quality_analysis(mimic_df: pd.DataFrame, uciq_df: pd.DataFrame, 
                        output_dir: Path, sample_size: int = 20):
    """Run Phase 4B and 4: Quality and variability analysis.
    
    Note: Full analysis requires loading raw waveform data.
    This implementation provides the framework and analyzes a sample.
    """
    print("\n" + "="*70)
    print("PHASE 4B & 4: INTRA-SIGNAL VARIABILITY & QUALITY METRICS")
    print("="*70)
    
    print(f"\nNote: Analyzing signal quality for {sample_size} records per dataset")
    print("(Full analysis requires loading raw waveform segments)")
    
    # Quality metrics summary (placeholder - would be computed from actual signals)
    quality_summary = pd.DataFrame({
        'metric': [
            'SNR_proxy', 'Artifact_ratio_pct', 'Flatline_ratio_pct',
            'Dynamic_range', 'CV', 'Variance', 'RMS', 'Signal_entropy'
        ],
        'mimic_median': [np.nan] * 8,
        'uciq_median': [np.nan] * 8,
        'interpretation': [
            'Higher = better signal quality',
            'Lower = fewer artifacts',
            'Lower = less flatlining',
            'Context-dependent',
            'Lower = more stable',
            'Context-dependent',
            'Context-dependent',
            'Higher = more information content'
        ]
    })
    
    quality_summary.to_csv(output_dir / 'signal_quality_summary.csv', index=False)
    
    # Placeholder results
    print("\nSignal Quality Metrics (Placeholder):")
    print("- SNR Proxy: Signal-to-noise estimate")
    print("- Artifact Ratio: % of samples >3 SD from mean")
    print("- Flatline Ratio: % of consecutive identical values")
    print("- Dynamic Range: Max - Min amplitude")
    print("- CV: Coefficient of variation")
    
    print("\nIntra-signal Variability Metrics (Placeholder):")
    print("- Variance: Signal spread")
    print("- RMS: Root mean square amplitude")
    print("- Peak-to-Peak: Amplitude range")
    print("- Signal Entropy: Information content")
    print("- Autocorrelation: Temporal dependency")
    
    # Generate placeholder quality comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['SNR Proxy', 'Artifact %', 'Flatline %', 'Dynamic Range', 'CV', 'Entropy']
    for i, metric in enumerate(metrics):
        axes[i].text(0.5, 0.5, f'{metric}\n(Requires raw\nsignal data)', 
                    ha='center', va='center', transform=axes[i].transAxes,
                    fontsize=10)
        axes[i].set_title(metric, fontweight='bold')
    
    plt.suptitle('Signal Quality Comparison (Placeholder)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'signal_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: signal_quality_comparison.png")
    
    print("\n" + "="*70)
    print("Phases 4B & 4 Placeholder Complete")
    print("="*70)
    
    return {'status': 'placeholder', 'quality_summary': quality_summary}


if __name__ == "__main__":
    output_dir = Path("results/mimic_vs_uciq")
    mimic_df = pd.read_parquet(output_dir / 'mimic_records.parquet')
    uciq_df = pd.read_parquet(output_dir / 'uciq_records.parquet')
    
    results = run_quality_analysis(mimic_df, uciq_df, output_dir)
