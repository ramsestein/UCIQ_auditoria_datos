"""
Análisis C: Detección de artefactos por tipo de señal.
Identifica valores fisiológicamente imposibles o sospechosos en señales numéricas clave.
Genera estadísticas de artefactos por señal y por box, y visualizaciones.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv, time

try:
    import vitaldb
except ImportError:
    vitaldb = None

OUTPUT_DIR = "results_auditory"
META_CSV = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")
OUT_CSV = os.path.join(OUTPUT_DIR, "artifact_detection_summary.csv")
OUT_DETAIL = os.path.join(OUTPUT_DIR, "artifact_detection_detail.csv")
MAX_WORKERS = 4
TIMEOUT = 45

# Límites fisiológicos para señales numéricas (tipo 2, 1 Hz)
# Formato: track_keyword -> (min_valid, max_valid, description)
PHYSIO_LIMITS = {
    'ECG_HR':       (20, 300, 'Heart Rate (ECG)'),
    'PLETH_HR':     (20, 300, 'Heart Rate (Pleth)'),
    'PLETH_SAT_O2': (30, 100, 'SpO2'),
    'ABP_SYS':      (30, 300, 'Arterial BP Systolic'),
    'ABP_DIA':      (10, 200, 'Arterial BP Diastolic'),
    'ABP_MEAN':     (15, 250, 'Arterial MAP'),
    'ART_SYS':      (30, 300, 'Art Line Systolic'),
    'ART_DIA':      (10, 200, 'Art Line Diastolic'),
    'ART_MEAN':     (15, 250, 'Art Line MAP'),
    'NIBP_SYS':     (40, 280, 'NIBP Systolic'),
    'NIBP_DIA':     (20, 180, 'NIBP Diastolic'),
    'NIBP_MEAN':    (25, 220, 'NIBP MAP'),
    'RR':           (2, 80, 'Respiratory Rate'),
    'BT_SKIN':      (25, 45, 'Skin Temperature'),
    'TEMP':         (25, 45, 'Temperature'),
    'ICP_MEAN':     (-10, 80, 'ICP Mean'),
    'FIO2':         (15, 100, 'FiO2'),
    'AWAY_CO2_ET':  (5, 100, 'EtCO2'),
}

# Spike: cambio entre muestras consecutivas que supera threshold
SPIKE_THRESHOLDS = {
    'ECG_HR':       80,
    'PLETH_HR':     80,
    'PLETH_SAT_O2': 30,
    'ABP_SYS':      80,
    'ABP_MEAN':     60,
    'ART_SYS':      80,
    'ART_MEAN':     60,
    'RR':           30,
}


def _match_track(track_name, keyword):
    """Verifica si un track_name contiene el keyword (case-insensitive)."""
    return keyword.lower() in track_name.lower() and '/' in track_name


def analyze_file(filepath):
    """Analiza un .vital y devuelve métricas de artefactos por señal."""
    if vitaldb is None:
        return []
    try:
        vf = vitaldb.VitalFile(str(filepath))
    except Exception:
        return []

    results = []
    trk_names = list(vf.trks.keys())

    for keyword, (lo, hi, desc) in PHYSIO_LIMITS.items():
        # Find matching track
        matched = [t for t in trk_names if _match_track(t, keyword)]
        if not matched:
            continue
        tname = matched[0]
        trk = vf.trks[tname]
        # Only numeric tracks (type 2) for physiological limits
        if trk.type != 2:
            continue
        try:
            df = vf.to_pandas(track_names=[tname], interval=1)
            if df is None or df.empty:
                continue
            col = df.columns[0]
            vals = df[col].dropna()
            n_total = len(vals)
            if n_total == 0:
                continue

            n_below = int((vals < lo).sum())
            n_above = int((vals > hi).sum())
            n_artifact = n_below + n_above
            pct_artifact = round(100.0 * n_artifact / n_total, 3)

            # Spike detection
            n_spikes = 0
            if keyword in SPIKE_THRESHOLDS:
                diffs = vals.diff().abs()
                n_spikes = int((diffs > SPIKE_THRESHOLDS[keyword]).sum())

            results.append({
                'signal': keyword,
                'signal_desc': desc,
                'n_total': n_total,
                'n_below_min': n_below,
                'n_above_max': n_above,
                'n_artifact': n_artifact,
                'pct_artifact': pct_artifact,
                'n_spikes': n_spikes,
                'mean_val': round(float(vals.mean()), 2),
                'std_val': round(float(vals.std()), 2),
            })
        except Exception:
            continue

    return results


def _worker(task):
    fname = task['filename']
    box = task['box']
    path = task['path']
    arts = analyze_file(path)
    for a in arts:
        a['filename'] = fname
        a['box'] = box
    return arts


def run():
    meta = pd.read_csv(META_CSV)
    tasks = []
    for _, row in meta.iterrows():
        p = Path(row['full_path'])
        if p.exists():
            tasks.append({
                'filename': row['filename'],
                'box': row['box'],
                'path': str(p),
            })

    print(f"Analyzing artifacts in {len(tasks)} files...")
    all_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result(timeout=TIMEOUT)
                all_rows.extend(res)
            except Exception:
                pass
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} files processed...")

    print(f"  {done}/{len(tasks)} files processed.")

    if not all_rows:
        print("No artifact data collected.")
        return

    detail = pd.DataFrame(all_rows)
    detail.to_csv(OUT_DETAIL, index=False)
    print(f"Detail CSV: {OUT_DETAIL} ({len(detail)} rows)")

    # Summary per signal
    summary = detail.groupby('signal').agg(
        signal_desc=('signal_desc', 'first'),
        n_files=('filename', 'nunique'),
        total_samples=('n_total', 'sum'),
        total_artifacts=('n_artifact', 'sum'),
        total_spikes=('n_spikes', 'sum'),
        mean_pct_artifact=('pct_artifact', 'mean'),
        median_pct_artifact=('pct_artifact', 'median'),
        p95_pct_artifact=('pct_artifact', lambda x: np.percentile(x, 95)),
        mean_signal_value=('mean_val', 'mean'),
    ).reset_index().sort_values('n_files', ascending=False)

    summary['overall_pct_artifact'] = (
        100.0 * summary['total_artifacts'] / summary['total_samples']
    ).round(3)

    summary.to_csv(OUT_CSV, index=False)
    print(f"Summary CSV: {OUT_CSV}")

    # Summary per signal per box
    box_summary = detail.groupby(['signal', 'box']).agg(
        n_files=('filename', 'nunique'),
        total_samples=('n_total', 'sum'),
        total_artifacts=('n_artifact', 'sum'),
        mean_pct_artifact=('pct_artifact', 'mean'),
    ).reset_index()
    box_summary['overall_pct_artifact'] = (
        100.0 * box_summary['total_artifacts'] / box_summary['total_samples']
    ).round(3)
    box_csv = os.path.join(OUTPUT_DIR, "artifact_by_box.csv")
    box_summary.to_csv(box_csv, index=False)

    # === Visualizaciones ===
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1) Bar chart: % artefactos por señal
    fig, ax = plt.subplots(figsize=(12, 7))
    sdata = summary.sort_values('overall_pct_artifact', ascending=True)
    colors = ['#e74c3c' if v > 5 else '#f39c12' if v > 1 else '#27ae60'
              for v in sdata['overall_pct_artifact']]
    bars = ax.barh(sdata['signal_desc'], sdata['overall_pct_artifact'], color=colors)
    ax.set_xlabel('% of Samples Outside Physiological Limits')
    ax.set_title('Artifact Rate by Signal Type')
    for bar, val, nf in zip(bars, sdata['overall_pct_artifact'], sdata['n_files']):
        label = f'{val:.2f}% (n={nf})'
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'viz_artifact_rates.png'), dpi=150)
    plt.close()

    # 2) Heatmap: artefactos por señal × box
    pivot = box_summary.pivot_table(
        index='signal', columns='box', values='overall_pct_artifact', fill_value=0
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, cbar_kws={'label': '% Artifacts'})
        ax.set_title('Artifact Rate (%) by Signal × Box')
        ax.set_ylabel('Signal')
        ax.set_xlabel('Box')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'viz_artifact_heatmap.png'), dpi=150)
        plt.close()

    # 3) Spike chart
    spike_data = summary[summary['total_spikes'] > 0].sort_values('total_spikes', ascending=True)
    if not spike_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(spike_data['signal_desc'], spike_data['total_spikes'], color='#8e44ad')
        ax.set_xlabel('Total Spikes Detected')
        ax.set_title('Sudden Value Spikes by Signal Type')
        for bar, val in zip(ax.patches, spike_data['total_spikes']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'viz_artifact_spikes.png'), dpi=150)
        plt.close()

    print("\nArtifact analysis complete. Visualizations saved.")
    print(summary[['signal_desc', 'n_files', 'overall_pct_artifact', 'total_spikes']].to_string(index=False))


if __name__ == '__main__':
    run()
