import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parents[0]
CSV = BASE / 'results_auditory' / 'clinical_metadata_audit.csv'
OUTDIR = BASE / 'results_auditory'

def run():
    if not CSV.exists():
        print('clinical_metadata_audit.csv not found')
        return
    df = pd.read_csv(CSV)
    # Ensure date parsed
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['month'] = df['date'].dt.to_period('M')

    # Signals columns detected by prefix has_
    cols = [c for c in df.columns if c.startswith('has_')]
    if not cols:
        print('No signal columns (has_*) found')
        return

    # Completeness per signal per month: percent of files in month that have the signal
    monthly = df.groupby('month')[cols].mean() * 100
    monthly.index = monthly.index.to_timestamp()
    monthly.to_csv(OUTDIR / 'completeness_by_month.csv')

    # Heatmap: months x signals
    plt.figure(figsize=(12, max(4, 0.4 * len(cols))))
    sns.heatmap(monthly.T, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label':'% presence'})
    plt.title('Completeness by Signal (percent of files per month with signal)')
    plt.ylabel('Signal')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig(OUTDIR / 'heatmap_completeness_by_month.png')
    plt.close()

    # Completeness per box overall
    box_comp = df.groupby('box')[cols].mean() * 100
    box_comp.to_csv(OUTDIR / 'completeness_by_box.csv')

    plt.figure(figsize=(12, max(6, 0.3 * len(box_comp.index))))
    sns.heatmap(box_comp, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label':'% presence'})
    plt.title('Completeness by Signal and Box (overall)')
    plt.ylabel('Box')
    plt.xlabel('Signal')
    plt.tight_layout()
    plt.savefig(OUTDIR / 'heatmap_completeness_by_box.png')
    plt.close()

    print('Completeness analysis finished. Outputs: completeness_by_month.csv, completeness_by_box.csv, heatmaps')

if __name__ == '__main__':
    run()
