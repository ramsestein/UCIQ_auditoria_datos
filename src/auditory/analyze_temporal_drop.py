import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parents[0]
CSV = BASE / 'results_auditory' / 'clinical_metadata_audit.csv'
OUTDIR = BASE / 'results_auditory'

def run():
    if not CSV.exists():
        print('clinical_metadata_audit.csv not found')
        return
    df = pd.read_csv(CSV)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    summary = []
    for box, g in df.groupby('box'):
        dates = g['date'].sort_values()
        # compute gaps in days between consecutive sessions
        diffs = dates.diff().dt.total_seconds().div(3600) # hours
        # ignore first NaN
        gaps = diffs.iloc[1:]
        # classify gaps
        micro = (gaps < 1).sum()
        short = ((gaps >=1) & (gaps < 24)).sum()
        long = ((gaps >=24) & (gaps < 168)).sum()
        very_long = (gaps >= 168).sum()
        median_gap = gaps.median()
        mean_gap = gaps.mean()
        summary.append({'box':box,'n_sessions':len(g),'median_gap_hours':median_gap,'mean_gap_hours':mean_gap,'micro':micro,'short':short,'long':long,'very_long':very_long})

    out = pd.DataFrame(summary)
    out.to_csv(OUTDIR / 'box_gap_summary.csv', index=False)

    # Aggregate gap distribution across all boxes
    all_diffs = df.groupby('box')['date'].apply(lambda s: s.sort_values().diff().dt.total_seconds().div(3600).dropna()).explode().astype(float)
    all_diffs = all_diffs.dropna()
    # basic stats
    stats = {
        'count': len(all_diffs),
        'median_hours': all_diffs.median(),
        'mean_hours': all_diffs.mean(),
        'p90_hours': all_diffs.quantile(0.9)
    }
    pd.Series(stats).to_csv(OUTDIR / 'gap_stats_overall.csv')

    print('Temporal drop analysis finished. Outputs: box_gap_summary.csv, gap_stats_overall.csv')

if __name__ == '__main__':
    run()
