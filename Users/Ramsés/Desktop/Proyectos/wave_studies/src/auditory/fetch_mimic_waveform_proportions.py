"""
Fetch signal-presence proportions for all 200 records in MIMIC-IV Waveform DB v0.1.0.

Uses the wfdb library to read headers directly from PhysioNet.
Local RECORDS files (already present for all 198 patients) provide record IDs,
avoiding HTTP directory listing. Only .hea headers are fetched remotely.
"""
import wfdb
import os
import re
import csv
import json
import time
import glob

PN_DIR = 'mimic4wdb/0.1.0'
WAVES_LOCAL = 'data/mimic4/waves'
OUT_CSV = 'results_auditory/mimic_waveform_proportions.csv'
PROG_JSON = 'results_auditory/_mimic_header_progress.json'


def collect_all_records():
    """Read local RECORDS files to build list of (pn_subdir, record_id) tuples."""
    records = []
    for rf in sorted(glob.glob(os.path.join(WAVES_LOCAL, 'p*', 'p*', 'RECORDS'))):
        patient_rel = os.path.relpath(os.path.dirname(rf), 'data/mimic4').replace('\\', '/')
        with open(rf, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # line like '83411188/83411188'
                rec_id = os.path.basename(line.rstrip('/'))
                pn_subdir = f"{PN_DIR}/{patient_rel}/{rec_id}"
                records.append((pn_subdir, rec_id))
    return records


def load_progress():
    """Load previously processed record IDs and their signals."""
    if not os.path.exists(PROG_JSON):
        return {}
    try:
        with open(PROG_JSON, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('records', {})
    except Exception:
        return {}


def save_progress(records_data):
    """Save processed records and their signals."""
    os.makedirs(os.path.dirname(PROG_JSON), exist_ok=True)
    with open(PROG_JSON, 'w', encoding='utf-8') as f:
        json.dump({'records': records_data}, f)


def get_signals_for_record(pn_subdir, rec_id, max_segments=3):
    """Read master header, then up to max_segments segment headers to find all signal names."""
    try:
        master = wfdb.rdheader(rec_id, pn_dir=pn_subdir)
    except Exception as e:
        print(f"  WARN: cannot read master header {rec_id}: {e}")
        return None

    # If single-segment record, sig_name is directly available
    if master.sig_name:
        return list(set(master.sig_name))

    # Multi-segment: read a few segment headers
    if not hasattr(master, 'seg_name') or not master.seg_name:
        return []

    signals = set()
    segments_read = 0
    for seg_name in master.seg_name:
        if seg_name == '~' or not seg_name:
            continue
        try:
            seg_hdr = wfdb.rdheader(seg_name, pn_dir=pn_subdir)
            if seg_hdr.sig_name:
                signals.update(seg_hdr.sig_name)
            segments_read += 1
            if segments_read >= max_segments:
                break
        except Exception:
            continue
        time.sleep(0.05)

    return list(signals)


def main():
    all_records = collect_all_records()
    print(f"Found {len(all_records)} records across all patients.")

    # Load previous progress
    progress = load_progress()
    print(f"Resuming: {len(progress)} records already processed.")

    failed = []
    for i, (pn_subdir, rec_id) in enumerate(all_records):
        if rec_id in progress:
            continue
        print(f"[{i+1}/{len(all_records)}] {rec_id} ...", end=' ', flush=True)
        signals = get_signals_for_record(pn_subdir, rec_id)
        if signals is None:
            failed.append(rec_id)
            print("FAILED")
            time.sleep(1)
            continue
        progress[rec_id] = signals
        print(f"OK ({len(signals)} signals: {', '.join(sorted(signals))})")
        save_progress(progress)
        time.sleep(0.1)

    # Compute proportions
    total = len(progress)
    signal_counts = {}
    for rec_id, sigs in progress.items():
        for s in sigs:
            signal_counts[s] = signal_counts.get(s, 0) + 1

    # Write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['signal', 'n_records_present', 'total_records', 'proportion'])
        for s, cnt in sorted(signal_counts.items(), key=lambda x: -x[1]):
            prop = cnt / total if total else 0
            writer.writerow([s, cnt, total, f"{prop:.4f}"])

    print(f"\nDone. {len(progress)} records processed, {len(failed)} failed.")
    if failed:
        print(f"Failed records: {failed}")
    print(f"Signals found: {len(signal_counts)}")
    print(f"Output: {OUT_CSV}")

if __name__ == '__main__':
    main()
