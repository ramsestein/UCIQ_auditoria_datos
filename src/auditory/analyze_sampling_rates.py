"""
Análisis B: Frecuencia de muestreo por tipo de track.
Muestrea ficheros para documentar las frecuencias reales (Hz) de cada señal.
Esencial para reproducibilidad: otros investigadores necesitan saber a qué Hz están las señales.
"""
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

try:
    import vitaldb
except ImportError:
    vitaldb = None

OUTPUT_DIR = "results_auditory"
CSV = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")
SAMPLES_PER_BOX = 15  # muestreo representativo
MAX_WORKERS = 4
TIMEOUT = 30


def get_track_rates(filepath):
    """Devuelve dict {track_name: sample_rate_hz} para un fichero .vital."""
    if vitaldb is None:
        return {}
    try:
        vf = vitaldb.VitalFile(str(filepath))
        rates = {}
        for tname, trk in vf.trks.items():
            try:
                sr = trk.srate
                if sr and sr > 0:
                    rates[tname] = round(sr, 2)
            except Exception:
                pass
        return rates
    except Exception:
        return {}


def _worker(task):
    fname = task['filename']
    path = task['path']
    box = task['box']
    rates = get_track_rates(path)
    return {'filename': fname, 'box': box, 'rates': rates}


def run():
    df = pd.read_csv(CSV)
    tasks = []
    for box, g in df.groupby('box'):
        n = min(SAMPLES_PER_BOX, len(g))
        samples = g.sample(n=n, random_state=42)
        for _, row in samples.iterrows():
            p = Path(row['full_path'])
            if p.exists():
                tasks.append({'filename': row['filename'], 'box': box, 'path': str(p)})

    print(f'Sampling {len(tasks)} files for track rates...')
    all_rates = defaultdict(list)  # track_name -> [hz values]
    rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result(timeout=TIMEOUT)
                for trk, hz in res.get('rates', {}).items():
                    # normalize track name to upper
                    norm = trk.upper().strip()
                    all_rates[norm].append(hz)
                    rows.append({'track': norm, 'box': res['box'],
                                 'filename': res['filename'], 'hz': hz})
            except Exception:
                pass
            if done % 20 == 0:
                print(f'  {done}/{len(tasks)}')

    # Save raw
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, 'sampling_rates_raw.csv'), index=False)

    # Summary per track
    summary = []
    for trk, hz_list in sorted(all_rates.items()):
        arr = np.array(hz_list)
        summary.append({
            'track': trk,
            'n_files': len(arr),
            'median_hz': np.median(arr),
            'mean_hz': np.mean(arr),
            'min_hz': np.min(arr),
            'max_hz': np.max(arr),
            'std_hz': np.std(arr),
        })
    sum_df = pd.DataFrame(summary).sort_values('n_files', ascending=False)
    sum_df.to_csv(os.path.join(OUTPUT_DIR, 'sampling_rates_summary.csv'), index=False)

    # Visualización: top 20 tracks por frecuencia
    import matplotlib.pyplot as plt
    top = sum_df.head(25)
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top['track'], top['median_hz'], color='teal', alpha=0.8)
    ax.set_xlabel('Median Sampling Rate (Hz)')
    ax.set_title('Sampling Rates by Signal Track (top 25)')
    ax.invert_yaxis()
    for bar, val in zip(bars, top['median_hz']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'viz_sampling_rates.png'), dpi=150)
    plt.close()

    print(f'Sampling rate analysis done. {len(summary)} unique tracks found.')


if __name__ == '__main__':
    run()
