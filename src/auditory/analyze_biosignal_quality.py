import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import csv

try:
    import vitaldb
except Exception:
    vitaldb = None

BASE = Path(__file__).parents[0]
CSV = BASE / 'results_auditory' / 'clinical_metadata_audit.csv'
OUTDIR = BASE / 'results_auditory'

# Number of files to sample per box. Set to None to process todos los ficheros por caja (más lento).
SAMPLES_PER_BOX = None
WINDOW_S = 60  # window length in seconds for quick checks


def analyze_file_quick(path):
    # Return simple metrics: pct_non_nan, std_est, flatline_pct
    if vitaldb is None:
        return {'pct_non_nan': None, 'std': None, 'flatline_pct': None}
    try:
        # Try opening the file, allow a couple retries for transient IO issues
        tries = 0
        vf = None
        while tries < 3:
            try:
                vf = vitaldb.VitalFile(str(path))
                break
            except Exception:
                tries += 1
                time.sleep(0.2)
        if vf is None:
            # couldn't open
            return {'pct_non_nan': None, 'std': None, 'flatline_pct': None}

        tracks = []
        try:
            tracks = vf.get_track_names()
        except Exception:
            tracks = []
        if not tracks:
            # Try reading all tracks into a DataFrame as a fallback
            try:
                df_all = vf.to_pandas(interval=5)
                if df_all is None or df_all.empty:
                    return {'pct_non_nan': 0, 'std': 0, 'flatline_pct': 1}
                # choose first non-empty column
                for c in df_all.columns:
                    if df_all[c].dropna().size > 0:
                        series = df_all[c].dropna()
                        break
                else:
                    return {'pct_non_nan': 0, 'std': 0, 'flatline_pct': 1}
                # estimate metrics from series
                pct_non_nan = len(series) / (len(df_all)) if len(df_all) > 0 else 0
                std = float(series.std())
                flatline_pct = 0.0
                return {'pct_non_nan': round(float(pct_non_nan*100),2), 'std': std, 'flatline_pct': round(float(flatline_pct*100),2)}
            except Exception:
                return {'pct_non_nan': 0, 'std': 0, 'flatline_pct': 1}
        # pick primary track (prefer pleth/ECG)
        preferred = None
        for pref in ['PLETH','SPO2','ECG','HR','ART']:
            for t in tracks:
                if pref.lower() in t.lower():
                    preferred = t
                    break
            if preferred:
                break
        if not preferred:
            preferred = tracks[0]
        # try multiple intervals to get a reasonable sample
        df = None
        for interval in (1, 5, 10):
            try:
                df = vf.to_pandas(track_names=[preferred], interval=interval)
                if df is not None and not df.empty:
                    break
            except Exception:
                df = None
        if df is None or df.empty:
            # as fallback try reading all tracks at coarse interval
            try:
                df_all = vf.to_pandas(interval=30)
                if df_all is None or df_all.empty:
                    return {'pct_non_nan': 0, 'std': 0, 'flatline_pct': 1}
                if preferred in df_all.columns:
                    series = df_all[preferred].dropna()
                else:
                    # pick first non-empty
                    non_empty_cols = [c for c in df_all.columns if df_all[c].dropna().size>0]
                    if not non_empty_cols:
                        return {'pct_non_nan': 0, 'std': 0, 'flatline_pct': 1}
                    series = df_all[non_empty_cols[0]].dropna()
                df = df_all
            except Exception:
                return {'pct_non_nan': None, 'std': None, 'flatline_pct': None}

        series = df[preferred].dropna() if preferred in df.columns else df.iloc[:,0].dropna()
        if series.empty:
            return {'pct_non_nan': 0, 'std': 0, 'flatline_pct': 1}
        pct_non_nan = len(series) / (len(df)) if len(df)>0 else 0
        std = float(series.std())
        # flatline percent: proportion of windows with near-zero std
        # calculate window based on sampling interval
        try:
            if len(df.index) > 1:
                dt = (df.index[1] - df.index[0]).total_seconds()
            else:
                dt = 1
            win = int(max(1, WINDOW_S / dt))
        except Exception:
            win = 1
        if win <= 1:
            flatline_pct = 1.0 if std == 0 else 0.0
        else:
            arr = series.values
            chunks = [arr[i:i+win] for i in range(0, len(arr), win) if len(arr[i:i+win]) == win]
            flats = sum(1 for c in chunks if np.std(c) < 1e-3)
            flatline_pct = flats / len(chunks) if chunks else 0
        return {'pct_non_nan': round(float(pct_non_nan*100),2), 'std': std, 'flatline_pct': round(float(flatline_pct*100),2)}
    except Exception as e:
        # return None so caller can distinguish failures from valid zero-content files
        return {'pct_non_nan': None, 'std': None, 'flatline_pct': None}


def run():
    if not CSV.exists():
        print('clinical_metadata_audit.csv not found')
        return
    df = pd.read_csv(CSV)
    out_rows = []
    # Resume support: read existing output and skip processed files
    processed = set()
    out_path = OUTDIR / 'quality_sample_summary.csv'
    if out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            processed = set(prev['filename'].astype(str).tolist())
        except Exception:
            processed = set()

    # build tasks list
    tasks = []
    for box, g in df.groupby('box'):
        if SAMPLES_PER_BOX is None or (isinstance(SAMPLES_PER_BOX, int) and SAMPLES_PER_BOX <= 0):
            n = len(g)
        else:
            n = min(SAMPLES_PER_BOX, len(g))
        try:
            samples = g.sample(n=n, random_state=1)
        except Exception:
            samples = g
        for _, row in samples.iterrows():
            fname = str(row['filename'])
            if fname in processed:
                continue
            p = Path(row.get('full_path', ''))
            tasks.append({'filename': fname, 'box': row.get('box'), 'path': str(p)})

    # process with a pool, per-file timeout
    max_workers = 4
    per_file_timeout = 30  # seconds
    total = len(tasks)
    if total == 0:
        print('No new files to process. Output up-to-date.')
        return

    print(f'Starting quality processing for {total} files with {max_workers} workers, timeout {per_file_timeout}s/file')

    # ensure outdir
    os.makedirs(OUTDIR, exist_ok=True)

    # write header if file doesn't exist
    if not out_path.exists():
        with open(out_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=['filename','box','pct_non_nan','std','flatline_pct'])
            writer.writeheader()

    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_proc_helper, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                res = fut.result(timeout=per_file_timeout)
            except TimeoutError:
                res = {'filename': task['filename'], 'box': task['box'], 'pct_non_nan': None, 'std': None, 'flatline_pct': None, 'error': 'timeout'}
            except Exception as e:
                res = {'filename': task['filename'], 'box': task['box'], 'pct_non_nan': None, 'std': None, 'flatline_pct': None, 'error': str(e)}

            # append to CSV incrementally
            row_out = {'filename': res.get('filename'), 'box': res.get('box'), 'pct_non_nan': res.get('pct_non_nan'), 'std': res.get('std'), 'flatline_pct': res.get('flatline_pct')}
            with open(out_path, 'a', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=['filename','box','pct_non_nan','std','flatline_pct'])
                writer.writerow(row_out)
            processed_count += 1
            if processed_count % 50 == 0 or processed_count == total:
                print(f'Processed {processed_count}/{total}')

    print(f'Quality processing finished. Appended {processed_count} rows to {out_path}')


def _proc_helper(task):
    """Helper executed in worker process to run analysis on a single file."""
    fname = task.get('filename')
    box = task.get('box')
    p = task.get('path')
    pth = Path(p)
    if not pth.exists():
        return {'filename': fname, 'box': box, 'pct_non_nan': None, 'std': None, 'flatline_pct': None}
    metrics = analyze_file_quick(pth)
    if metrics is None:
        metrics = {'pct_non_nan': None, 'std': None, 'flatline_pct': None}
    return {'filename': fname, 'box': box, 'pct_non_nan': metrics.get('pct_non_nan'), 'std': metrics.get('std'), 'flatline_pct': metrics.get('flatline_pct')}

if __name__ == '__main__':
    run()
