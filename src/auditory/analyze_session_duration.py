"""
Análisis A: Distribución de duraciones de sesión de monitorización.
Genera histograma, boxplot por box, y estadísticas descriptivas.
Permite a revisores entender la fragmentación real de los datos.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "results_auditory"
CSV = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")


def run():
    df = pd.read_csv(CSV)
    dur = df['duration_min'].dropna()

    stats = {
        'n_sessions': len(dur),
        'mean_min': dur.mean(),
        'median_min': dur.median(),
        'std_min': dur.std(),
        'p5_min': dur.quantile(0.05),
        'p25_min': dur.quantile(0.25),
        'p75_min': dur.quantile(0.75),
        'p95_min': dur.quantile(0.95),
        'max_min': dur.max(),
        'min_min': dur.min(),
        'pct_micro_lt5': (dur < 5).mean() * 100,
        'pct_short_5_30': ((dur >= 5) & (dur < 30)).mean() * 100,
        'pct_standard_30_60': ((dur >= 30) & (dur < 60)).mean() * 100,
        'pct_long_60_240': ((dur >= 60) & (dur < 240)).mean() * 100,
        'pct_vlong_gt240': (dur >= 240).mean() * 100,
    }
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'session_duration_stats.csv'), index=False)

    # --- Histograma general ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma con bins logarítmicos para cubrir micro y largas
    bins = np.concatenate([[0, 1, 5, 15, 30, 60, 120, 240, 480], [dur.max() + 1]])
    axes[0].hist(dur, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(dur.median(), color='red', ls='--', label=f'Median={dur.median():.0f}m')
    axes[0].axvline(dur.mean(), color='orange', ls='--', label=f'Mean={dur.mean():.0f}m')
    axes[0].set_xlabel('Duration (min)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Session Duration Distribution')
    axes[0].legend()

    # Barras por categoría
    cats = ['<5m', '5-30m', '30-60m', '1-4h', '>4h']
    vals = [stats['pct_micro_lt5'], stats['pct_short_5_30'],
            stats['pct_standard_30_60'], stats['pct_long_60_240'],
            stats['pct_vlong_gt240']]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
    axes[1].bar(cats, vals, color=colors, edgecolor='white')
    axes[1].set_ylabel('% of sessions')
    axes[1].set_title('Session Duration Categories')
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'viz_session_duration_distribution.png'), dpi=150)
    plt.close()

    # --- Boxplot por box ---
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby('box')['duration_min'].median().sort_values().index
    sns.boxplot(data=df, x='box', y='duration_min', order=order, ax=ax,
                palette='Set2', showfliers=False)
    sns.stripplot(data=df, x='box', y='duration_min', order=order, ax=ax,
                  color='black', alpha=0.1, size=2)
    ax.set_ylabel('Duration (min)')
    ax.set_title('Session Duration by Box (outliers hidden)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'viz_session_duration_by_box.png'), dpi=150)
    plt.close()

    print(f'Session duration analysis done.')
    print(f'  Mean: {dur.mean():.1f}m, Median: {dur.median():.1f}m, Std: {dur.std():.1f}m')
    print(f'  Micro(<5m): {stats["pct_micro_lt5"]:.1f}%, Long(>4h): {stats["pct_vlong_gt240"]:.1f}%')


if __name__ == '__main__':
    run()
