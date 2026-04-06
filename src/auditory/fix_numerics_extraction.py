"""
================================================================================
FIX: MIMIC NUMERICS CSV + UCIQ VITALDB NUMERIC EXTRACTION
================================================================================

PROBLEM 1 - MIMIC:
  The local WFDB files only contain waveforms (ECG 250Hz, Pleth 125Hz, Resp 62Hz).
  The numeric trends (HR, SpO2, RR, ABP values at ~1Hz) are in SEPARATE files:
    {record_id}_numerics.csv.gz
  These must be downloaded from PhysioNet for each record.

PROBLEM 2 - UCIQ:
  vitaldb.VitalFile().to_numpy() returns NaN for numeric tracks.
  This is a known issue with certain .vital file versions.
  Solution: use vitaldb's track iteration or manual binary parsing.

================================================================================
"""

import os
import sys
import gzip
import time
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO, BytesIO
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: MIMIC NUMERICS — DOWNLOAD CSV FROM PHYSIONET
# ============================================================================

def get_mimic_record_list():
    """
    Get the list of all 200 records in MIMIC-IV Waveform v0.1.0.
    Returns list of dicts with subject_dir and record_id.
    """
    import wfdb
    
    database = 'mimic4wdb/0.1.0'
    subjects = wfdb.get_record_list(database)
    
    records = []
    for subject in subjects:
        studies = wfdb.get_record_list(f'{database}/{subject}')
        for study in studies:
            record_id = study.split('/')[-1] if '/' in study else study
            records.append({
                'subject_dir': subject,
                'record_id': record_id,
            })
    
    print(f"Found {len(records)} MIMIC records")
    return records


def download_mimic_numerics_csv(record_id: str, subject_dir: str, 
                                 cache_dir: str = './mimic_numerics_cache') -> pd.DataFrame:
    """
    Download the numerics CSV file for a single MIMIC record using wfdb.
    """
    import urllib.request
    import urllib.error
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{record_id}_numerics.csv')
    
    # Check cache first
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            if len(df) > 0:
                return df
        except:
            pass
    
    # Try local files first - check if _n.csv.gz or _numerics.csv.gz exists locally
    mimic_base = 'data/mimic4wdb_full'
    for root in [f'{mimic_base}/waves/{subject_dir}', f'{mimic_base}/{subject_dir}']:
        for suffix in ['_n', '_numerics']:
            for ext in ['.csv', '.csv.gz']:
                local_file = os.path.join(root, f'{record_id}{suffix}{ext}')
                if os.path.exists(local_file):
                    try:
                        if ext == '.csv.gz':
                            with gzip.open(local_file, 'rt') as f:
                                df = pd.read_csv(f)
                        else:
                            df = pd.read_csv(local_file)
                        if len(df) > 0:
                            df.to_csv(cache_file, index=False)
                            return df
                    except:
                        pass
    
    # Try using wfdb to read numerics channels from local segments
    try:
        import wfdb
        # Try to read any numeric channels from the waveform segments
        for root in [f'{mimic_base}/waves/{subject_dir}/{record_id}', 
                     f'{mimic_base}/{subject_dir}/{record_id}']:
            if os.path.exists(root):
                # Look for segment files
                seg_files = [f for f in os.listdir(root) if f.endswith('.hea') and '_' in f]
                if seg_files:
                    all_data = []
                    for seg_file in sorted(seg_files)[:20]:  # Limit to first 20 segments
                        try:
                            seg_path = os.path.join(root, seg_file.replace('.hea', ''))
                            rec = wfdb.rdrecord(seg_path)
                            if rec and hasattr(rec, 'p_signal') and rec.p_signal is not None:
                                # Look for numeric channels
                                numeric_channels = []
                                for i, name in enumerate(rec.sig_name):
                                    if name in ['HR', 'PULSE', 'SpO2', 'RR', 'RESP', 
                                               'ABP', 'ABPm', 'ABPs', 'ABPd', 'PAP']:
                                        numeric_channels.append((i, name))
                                
                                if numeric_channels:
                                    seg_data = {'Time': np.arange(len(rec.p_signal)) / rec.fs}
                                    for idx, name in numeric_channels:
                                        seg_data[name] = rec.p_signal[:, idx]
                                    all_data.append(pd.DataFrame(seg_data))
                        except:
                            continue
                    
                    if all_data:
                        df = pd.concat(all_data, ignore_index=True)
                        if len(df) > 10:
                            df.to_csv(cache_file, index=False)
                            return df
    except:
        pass
    
    # Download from PhysioNet as last resort
    subject_clean = subject_dir.rstrip('/')
    if subject_clean.startswith('waves/'):
        subject_clean = subject_clean[6:]
    
    base_urls = [
        f"https://physionet.org/files/mimic4wdb/0.1.0/waves/{subject_clean}/{record_id}/{record_id}n.csv.gz",
        f"https://physionet.org/files/mimic4wdb/0.1.0/{subject_clean}/{record_id}/{record_id}n.csv.gz",
    ]
    
    for url in base_urls:
        try:
            req = urllib.request.Request(url)
            # PhysioNet may require credentials for some datasets
            # MIMIC-IV Waveform v0.1.0 is open access
            response = urllib.request.urlopen(req, timeout=60)
            compressed_data = response.read()
            
            # Decompress
            csv_text = gzip.decompress(compressed_data).decode('utf-8', errors='replace')
            
            # Parse CSV
            df = pd.read_csv(StringIO(csv_text))
            
            # Cache locally
            df.to_csv(cache_file, index=False)
            
            return df
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Try next URL
                continue
            else:
                print(f"  HTTP Error {e.code} for {record_id}: {e.reason}")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Error downloading {record_id}: {e}")
            return pd.DataFrame()
    
    # All URLs failed
    return pd.DataFrame()


def extract_all_mimic_numerics(output_dir: str = './phase_outputs',
                                cache_dir: str = './mimic_numerics_cache',
                                max_records: int = 200) -> pd.DataFrame:
    """
    Download and summarize numerics for all MIMIC-IV Waveform records.
    """
    records = get_mimic_record_list()[:max_records]
    
    # Physiological plausibility ranges
    RANGES = {
        'HR': (20, 300), 'SpO2': (50, 100), 'PULSE': (20, 300),
        'RR': (2, 60),
        'ABPs': (30, 300), 'ABPd': (10, 200), 'ABPm': (20, 250),
        'NBPs': (30, 300), 'NBPd': (10, 200), 'NBPm': (20, 250),
    }
    
    # Clinical thresholds
    THRESHOLDS = {
        'HR': {'bradycardia': (0, 60), 'normal': (60, 100), 'tachycardia': (100, 999)},
        'SpO2': {'severe_hypoxemia': (0, 90), 'mild_hypoxemia': (90, 95), 'normal': (95, 101)},
        'ABPm': {'below_target': (0, 65), 'target': (65, 85), 'above_target': (85, 999)},
        'ABPs': {'hypotension': (0, 90), 'normal': (90, 140), 'hypertension': (140, 999)},
    }
    
    # Standardize column names
    COL_MAP = {
        'HR': 'HR', 'SpO2': 'SpO2', 'PULSE': 'PULSE',
        'RR': 'RR', 'RESP': 'RR',
        'ABPs': 'ABPs', 'ABPSys': 'ABPs', 'ABP Sys': 'ABPs',
        'ABPd': 'ABPd', 'ABPDias': 'ABPd', 'ABP Dias': 'ABPd',
        'ABPm': 'ABPm', 'ABPMean': 'ABPm', 'ABP Mean': 'ABPm',
        'NBPs': 'NBPs', 'NBPSys': 'NBPs', 'NBP Sys': 'NBPs',
        'NBPd': 'NBPd', 'NBPDias': 'NBPd', 'NBP Dias': 'NBPd',
        'NBPm': 'NBPm', 'NBPMean': 'NBPm', 'NBP Mean': 'NBPm',
    }
    
    summaries = []
    success_count = 0
    
    print(f"Downloading numerics for {len(records)} MIMIC records...")
    print(f"Cache directory: {cache_dir}")
    
    for i, rec in enumerate(records):
        if i % 25 == 0:
            print(f"  [{i+1}/{len(records)}] Downloading... ({success_count} successful so far)")
        
        df = download_mimic_numerics_csv(rec['record_id'], rec['subject_dir'], cache_dir)
        
        if df is None or len(df) < 10:
            summaries.append({
                'record_id': rec['record_id'],
                'dataset': 'MIMIC',
                'status': 'no_numerics',
                'n_rows': 0,
            })
            continue
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
        
        # Build summary
        summary = {
            'record_id': rec['record_id'],
            'dataset': 'MIMIC',
            'status': 'ok',
            'n_rows': len(df),
            'columns_found': ','.join([c for c in df.columns if c != 'Time']),
        }
        
        for vital in ['HR', 'SpO2', 'RR', 'PULSE', 'ABPm', 'ABPs', 'ABPd', 'NBPm', 'NBPs', 'NBPd']:
            if vital not in df.columns:
                continue
            
            values = df[vital].dropna()
            
            # Apply physiological range filter
            if vital in RANGES:
                lo, hi = RANGES[vital]
                values = values[(values >= lo) & (values <= hi)]
            
            if len(values) < 10:
                continue
            
            summary[f'{vital}_n'] = int(len(values))
            summary[f'{vital}_mean'] = round(float(np.mean(values)), 2)
            summary[f'{vital}_median'] = round(float(np.median(values)), 2)
            summary[f'{vital}_sd'] = round(float(np.std(values)), 2)
            summary[f'{vital}_cv'] = round(float(np.std(values) / np.mean(values)), 4) if np.mean(values) > 0 else np.nan
            summary[f'{vital}_p5'] = round(float(np.percentile(values, 5)), 2)
            summary[f'{vital}_p25'] = round(float(np.percentile(values, 25)), 2)
            summary[f'{vital}_p75'] = round(float(np.percentile(values, 75)), 2)
            summary[f'{vital}_p95'] = round(float(np.percentile(values, 95)), 2)
            summary[f'{vital}_min'] = round(float(np.min(values)), 2)
            summary[f'{vital}_max'] = round(float(np.max(values)), 2)
            
            # Clinical thresholds
            if vital in THRESHOLDS:
                for range_name, (lo_t, hi_t) in THRESHOLDS[vital].items():
                    pct = float(((values >= lo_t) & (values < hi_t)).mean())
                    summary[f'{vital}_pct_{range_name}'] = round(pct, 4)
            
            # Event counts (for Phase 4B)
            if vital == 'ABPm' and len(values) > 300:
                # Hypotension: MAP < 65 for >=5 min (>=300 consecutive samples at ~1Hz)
                below_65 = (values < 65).values
                max_run = 0
                current = 0
                n_events = 0
                for v in below_65:
                    if v:
                        current += 1
                    else:
                        if current >= 300:
                            n_events += 1
                        max_run = max(max_run, current)
                        current = 0
                if current >= 300:
                    n_events += 1
                
                duration_hours = len(values) / 3600
                summary['hypotension_events'] = n_events
                summary['hypotension_rate_per_hour'] = round(n_events / duration_hours, 4) if duration_hours > 0 else 0
            
            if vital == 'HR' and len(values) > 300:
                # Tachycardia: HR > 120 for >=5 min
                above_120 = (values > 120).values
                current = 0
                n_events = 0
                for v in above_120:
                    if v:
                        current += 1
                    else:
                        if current >= 300:
                            n_events += 1
                        current = 0
                if current >= 300:
                    n_events += 1
                
                duration_hours = len(values) / 3600
                summary['tachycardia_events'] = n_events
                summary['tachycardia_rate_per_hour'] = round(n_events / duration_hours, 4) if duration_hours > 0 else 0
            
            if vital == 'SpO2' and len(values) > 60:
                # Desaturation: SpO2 < 90 for >=1 min
                below_90 = (values < 90).values
                current = 0
                n_events = 0
                for v in below_90:
                    if v:
                        current += 1
                    else:
                        if current >= 60:
                            n_events += 1
                        current = 0
                if current >= 60:
                    n_events += 1
                
                duration_hours = len(values) / 3600
                summary['desaturation_events'] = n_events
                summary['desaturation_rate_per_hour'] = round(n_events / duration_hours, 4) if duration_hours > 0 else 0
        
        if summary.get('HR_n', 0) > 0 or summary.get('SpO2_n', 0) > 0:
            success_count += 1
        
        summaries.append(summary)
        
        # Rate limiting: be nice to PhysioNet
        time.sleep(0.1)
    
    result = pd.DataFrame(summaries)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, 'mimic_numerics_summary.csv'), index=False)
    result.to_parquet(os.path.join(output_dir, 'mimic_numerics_summary.parquet'))
    
    print(f"\nDone! {success_count}/{len(records)} records with numeric data")
    print(f"Saved to {output_dir}/mimic_numerics_summary.csv")
    
    # Quick summary
    for vital in ['HR', 'SpO2', 'RR', 'ABPm']:
        col = f'{vital}_n'
        n_with = (result[col] > 0).sum() if col in result.columns else 0
        print(f"  {vital}: {n_with}/{len(records)} records ({100*n_with/len(records):.0f}%)")
    
    return result


# ============================================================================
# PART 2: UCIQ NUMERICS — ALTERNATIVE VITALDB EXTRACTION
# ============================================================================

def extract_uciq_numeric_tracks(vital_file: str) -> dict:
    """
    Extract numeric 1Hz tracks from a .vital file using multiple methods.
    
    Method 1: vitaldb with explicit track loading
    Method 2: vitaldb with to_pandas
    Method 3: Manual parsing of track metadata
    
    Returns dict of {signal_name: numpy_array} or empty dict if all fail.
    """
    
    TRACK_MAP = {
        'ECG_HR': 'HR', 'PLETH_HR': 'HR', 'HR': 'HR',
        'PLETH_SAT_O2': 'SpO2', 'SAT_O2': 'SpO2', 'SpO2': 'SpO2',
        'RR': 'RR',
        'ABP_MEAN': 'ABPm', 'ART_MEAN': 'ABPm',
        'ABP_SYS': 'ABPs', 'ART_SYS': 'ABPs',
        'ABP_DIA': 'ABPd', 'ART_DIA': 'ABPd',
        'NIBP_MEAN': 'NBPm', 'NIBP_SYS': 'NBPs', 'NIBP_DIA': 'NBPd',
        'ICP_MEAN': 'ICPm',
        'TEMP': 'Temp', 'BT_SKIN': 'TempSkin',
    }
    
    result = {}
    
    # ---- Method 1: vitaldb with explicit interval ----
    try:
        import vitaldb
        vf = vitaldb.VitalFile(vital_file)
        
        # Get available track names using get_track_names() method
        track_names = []
        if hasattr(vf, 'get_track_names'):
            track_names = vf.get_track_names()
        elif hasattr(vf, 'trks') and isinstance(vf.trks, list):
            # If trks is list of dicts, extract names
            for t in vf.trks:
                if isinstance(t, dict) and 'name' in t:
                    track_names.append(t['name'])
                elif isinstance(t, str):
                    track_names.append(t)
        elif hasattr(vf, 'tracks') and isinstance(vf.tracks, dict):
            track_names = list(vf.tracks.keys())
        
        for track_name in track_names:
            std_name = TRACK_MAP.get(track_name)
            if std_name is None:
                continue
            if std_name in result:
                continue  # already have this signal
            
            try:
                # Method 1a: to_numpy with explicit interval=1 (1 second)
                values = vf.to_numpy(track_name, 1.0)
                if values is not None and len(values) > 0:
                    n_valid = np.count_nonzero(~np.isnan(values))
                    if n_valid > 50:
                        result[std_name] = values
                        continue
            except:
                pass
            
            try:
                # Method 1b: to_numpy without interval (native rate)
                values = vf.to_numpy(track_name)
                if values is not None and len(values) > 0:
                    n_valid = np.count_nonzero(~np.isnan(values))
                    if n_valid > 50:
                        result[std_name] = values
                        continue
            except:
                pass
        
        if result:
            return result
            
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # ---- Method 2: vitaldb to_pandas ----
    try:
        import vitaldb
        vf = vitaldb.VitalFile(vital_file)
        
        target_tracks = [t for t in TRACK_MAP.keys()]
        
        # Try to_pandas with specific tracks
        for track_name in target_tracks:
            try:
                df = vf.to_pandas([track_name], 1.0)
                if df is not None and len(df) > 0 and track_name in df.columns:
                    values = df[track_name].values
                    n_valid = np.count_nonzero(~np.isnan(values))
                    if n_valid > 50:
                        std_name = TRACK_MAP[track_name]
                        if std_name not in result:
                            result[std_name] = values
            except:
                continue
        
        if result:
            return result
            
    except:
        pass
    
    # ---- Method 3: Read track info and try different loading ----
    try:
        import vitaldb
        vf = vitaldb.VitalFile(vital_file)
        
        # Some versions of vitaldb store tracks differently
        if hasattr(vf, 'trks'):
            for trk in vf.trks:
                if isinstance(trk, dict):
                    name = trk.get('name', '')
                    std_name = TRACK_MAP.get(name)
                    if std_name and std_name not in result:
                        # Try getting data from track object
                        if 'recs' in trk and trk['recs']:
                            try:
                                all_vals = []
                                for rec in trk['recs']:
                                    if isinstance(rec, dict) and 'val' in rec:
                                        all_vals.extend(rec['val'] if isinstance(rec['val'], list) else [rec['val']])
                                if all_vals:
                                    arr = np.array(all_vals, dtype=float)
                                    if np.count_nonzero(~np.isnan(arr)) > 50:
                                        result[std_name] = arr
                            except:
                                continue
    except:
        pass
    
    return result


def extract_all_uciq_numerics(vital_dir: str, output_dir: str = './phase_outputs',
                               max_files: int = 1000) -> pd.DataFrame:
    """
    Extract numeric summaries from UCIQ .vital files.
    """
    RANGES = {
        'HR': (20, 300), 'SpO2': (50, 100), 'RR': (2, 60),
        'ABPm': (20, 250), 'ABPs': (30, 300), 'ABPd': (10, 200),
        'NBPm': (20, 250), 'ICPm': (-10, 80), 'Temp': (30, 42),
    }
    
    THRESHOLDS = {
        'HR': {'bradycardia': (0, 60), 'normal': (60, 100), 'tachycardia': (100, 999)},
        'SpO2': {'severe_hypoxemia': (0, 90), 'mild_hypoxemia': (90, 95), 'normal': (95, 101)},
        'ABPm': {'below_target': (0, 65), 'target': (65, 85), 'above_target': (85, 999)},
        'ABPs': {'hypotension': (0, 90), 'normal': (90, 140), 'hypertension': (140, 999)},
    }
    
    vital_files = sorted(Path(vital_dir).glob('**/*.vital'))[:max_files]
    print(f"Found {len(vital_files)} .vital files in {vital_dir}")
    
    summaries = []
    success_count = 0
    method_counts = {'method1': 0, 'method2': 0, 'method3': 0, 'failed': 0}
    
    # Test first file to find working method
    if vital_files:
        print("Testing extraction methods on first file...")
        test_result = extract_uciq_numeric_tracks(str(vital_files[0]))
        if test_result:
            print(f"  Success! Found signals: {list(test_result.keys())}")
            for k, v in test_result.items():
                valid = np.count_nonzero(~np.isnan(v))
                print(f"    {k}: {valid}/{len(v)} valid samples ({100*valid/len(v):.1f}%)")
        else:
            print("  WARNING: No numeric data extracted from first file")
            print("  Trying different approach...")
            
            # Debug: print what's in the file
            try:
                import vitaldb
                vf = vitaldb.VitalFile(str(vital_files[0]))
                print(f"  File loaded. Type: {type(vf)}")
                if hasattr(vf, 'trks'):
                    print(f"  Tracks ({len(vf.trks)}):")
                    for trk in vf.trks[:20]:
                        if isinstance(trk, dict):
                            name = trk.get('name', '?')
                            freq = trk.get('srate', trk.get('freq', '?'))
                            n_recs = len(trk.get('recs', []))
                            print(f"    {name}: freq={freq}, n_recs={n_recs}")
                        else:
                            print(f"    {type(trk)}: {str(trk)[:100]}")
                elif hasattr(vf, 'tracks'):
                    print(f"  Tracks: {list(vf.tracks.keys()) if isinstance(vf.tracks, dict) else type(vf.tracks)}")
                else:
                    print(f"  Attributes: {[a for a in dir(vf) if not a.startswith('_')]}")
            except Exception as e:
                print(f"  Debug error: {e}")
    
    print(f"\nExtracting numerics from {len(vital_files)} files...")
    
    for i, vf_path in enumerate(vital_files):
        if i % 100 == 0:
            print(f"  [{i+1}/{len(vital_files)}] Processing... ({success_count} successful)")
        
        tracks = extract_uciq_numeric_tracks(str(vf_path))
        
        summary = {
            'record_id': vf_path.stem,
            'dataset': 'UCIQ',
            'n_signals_extracted': len(tracks),
        }
        
        if not tracks:
            summary['status'] = 'no_data'
            method_counts['failed'] += 1
            summaries.append(summary)
            continue
        
        summary['status'] = 'ok'
        summary['signals_found'] = ','.join(tracks.keys())
        success_count += 1
        
        for vital, values in tracks.items():
            # Clean
            clean = values[~np.isnan(values)]
            if vital in RANGES:
                lo, hi = RANGES[vital]
                clean = clean[(clean >= lo) & (clean <= hi)]
            
            if len(clean) < 10:
                continue
            
            summary[f'{vital}_n'] = int(len(clean))
            summary[f'{vital}_mean'] = round(float(np.mean(clean)), 2)
            summary[f'{vital}_median'] = round(float(np.median(clean)), 2)
            summary[f'{vital}_sd'] = round(float(np.std(clean)), 2)
            summary[f'{vital}_cv'] = round(float(np.std(clean) / np.mean(clean)), 4) if np.mean(clean) > 0 else np.nan
            summary[f'{vital}_p5'] = round(float(np.percentile(clean, 5)), 2)
            summary[f'{vital}_p25'] = round(float(np.percentile(clean, 25)), 2)
            summary[f'{vital}_p75'] = round(float(np.percentile(clean, 75)), 2)
            summary[f'{vital}_p95'] = round(float(np.percentile(clean, 95)), 2)
            
            if vital in THRESHOLDS:
                for range_name, (lo_t, hi_t) in THRESHOLDS[vital].items():
                    pct = float(((clean >= lo_t) & (clean < hi_t)).mean())
                    summary[f'{vital}_pct_{range_name}'] = round(pct, 4)
            
            # Events
            if vital == 'ABPm' and len(clean) > 300:
                below_65 = (clean < 65)
                current = 0
                n_events = 0
                for v in below_65:
                    if v: current += 1
                    else:
                        if current >= 300: n_events += 1
                        current = 0
                if current >= 300: n_events += 1
                dur_h = len(clean) / 3600
                summary['hypotension_events'] = n_events
                summary['hypotension_rate_per_hour'] = round(n_events / dur_h, 4) if dur_h > 0 else 0
            
            if vital == 'HR' and len(clean) > 300:
                above_120 = (clean > 120)
                current = 0
                n_events = 0
                for v in above_120:
                    if v: current += 1
                    else:
                        if current >= 300: n_events += 1
                        current = 0
                if current >= 300: n_events += 1
                dur_h = len(clean) / 3600
                summary['tachycardia_events'] = n_events
                summary['tachycardia_rate_per_hour'] = round(n_events / dur_h, 4) if dur_h > 0 else 0
            
            if vital == 'SpO2' and len(clean) > 60:
                below_90 = (clean < 90)
                current = 0
                n_events = 0
                for v in below_90:
                    if v: current += 1
                    else:
                        if current >= 60: n_events += 1
                        current = 0
                if current >= 60: n_events += 1
                dur_h = len(clean) / 3600
                summary['desaturation_events'] = n_events
                summary['desaturation_rate_per_hour'] = round(n_events / dur_h, 4) if dur_h > 0 else 0
        
        summaries.append(summary)
    
    result = pd.DataFrame(summaries)
    
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, 'uciq_numerics_summary.csv'), index=False)
    result.to_parquet(os.path.join(output_dir, 'uciq_numerics_summary.parquet'))
    
    print(f"\nDone! {success_count}/{len(vital_files)} files with numeric data")
    print(f"Method counts: {method_counts}")
    
    for vital in ['HR', 'SpO2', 'RR', 'ABPm']:
        col = f'{vital}_n'
        n_with = (result[col] > 0).sum() if col in result.columns else 0
        print(f"  {vital}: {n_with}/{len(vital_files)} files ({100*n_with/len(vital_files):.0f}%)")
    
    return result


# ============================================================================
# PART 3: QUICK DIAGNOSTIC — RUN THIS FIRST
# ============================================================================

def diagnose_uciq_file(vital_file: str):
    """
    Run this on a single .vital file to understand what's inside
    and why numeric extraction might be failing.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {vital_file}")
    print(f"{'='*60}")
    
    try:
        import vitaldb
        vf = vitaldb.VitalFile(vital_file)
        print(f"✅ File loaded successfully")
        
        # Print all attributes
        attrs = [a for a in dir(vf) if not a.startswith('_')]
        print(f"\nAttributes: {attrs}")
        
        # Try to get track info
        if hasattr(vf, 'get_track_names'):
            track_names = vf.get_track_names()
            print(f"\nvf.get_track_names(): {len(track_names)} tracks")
            for i, name in enumerate(track_names[:30]):
                print(f"  [{i}] {name}")
        elif hasattr(vf, 'trks'):
            print(f"\nvf.trks: {len(vf.trks)} tracks")
            for i, trk in enumerate(vf.trks):
                if isinstance(trk, dict):
                    name = trk.get('name', '?')
                    srate = trk.get('srate', trk.get('freq', trk.get('dt', '?')))
                    unit = trk.get('unit', '?')
                    n_recs = len(trk.get('recs', []))
                    print(f"  [{i}] {name}: srate={srate}, unit={unit}, recs={n_recs}")
                    
                    # Try to read some data
                    if n_recs > 0 and isinstance(trk['recs'], list):
                        first_rec = trk['recs'][0]
                        if isinstance(first_rec, dict):
                            print(f"       First rec keys: {list(first_rec.keys())}")
                            if 'val' in first_rec:
                                val = first_rec['val']
                                if isinstance(val, (list, np.ndarray)):
                                    arr = np.array(val[:10], dtype=float)
                                    print(f"       First 10 values: {arr}")
                                else:
                                    print(f"       Value type: {type(val)}, val={val}")
                else:
                    print(f"  [{i}] Type: {type(trk)}")
        
        # Try different loading methods
        print("\n--- Testing to_numpy ---")
        test_tracks = ['ECG_HR', 'PLETH_SAT_O2', 'RR', 'ABP_MEAN', 'HR', 'SpO2']
        for tn in test_tracks:
            try:
                vals = vf.to_numpy(tn, 1.0)
                if vals is not None:
                    n_valid = np.count_nonzero(~np.isnan(vals))
                    print(f"  {tn} (interval=1.0): len={len(vals)}, valid={n_valid}, first_valid={vals[~np.isnan(vals)][:3] if n_valid > 0 else 'none'}")
                else:
                    print(f"  {tn} (interval=1.0): None")
            except Exception as e:
                print(f"  {tn} (interval=1.0): ERROR - {e}")
            
            try:
                vals = vf.to_numpy(tn)
                if vals is not None:
                    n_valid = np.count_nonzero(~np.isnan(vals))
                    print(f"  {tn} (no interval): len={len(vals)}, valid={n_valid}")
                else:
                    print(f"  {tn} (no interval): None")
            except Exception as e:
                print(f"  {tn} (no interval): ERROR - {e}")
        
        print("\n--- Testing to_pandas ---")
        try:
            df = vf.to_pandas(['ECG_HR', 'PLETH_SAT_O2'], 1.0)
            if df is not None:
                print(f"  to_pandas result: {df.shape}, columns={list(df.columns)}")
                print(f"  Non-null counts:\n{df.count()}")
            else:
                print(f"  to_pandas returned None")
        except Exception as e:
            print(f"  to_pandas ERROR: {e}")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")


def diagnose_mimic_record(record_id: str, subject_dir: str):
    """
    Test MIMIC numerics CSV download for a single record.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSING MIMIC: {record_id} in {subject_dir}")
    print(f"{'='*60}")
    
    df = download_mimic_numerics_csv(record_id, subject_dir, cache_dir='./test_cache')
    
    if df is not None and len(df) > 0:
        print(f"✅ Downloaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nNon-null counts:")
        print(df.count())
        print(f"\nBasic stats:")
        print(df.describe())
    else:
        print(f"❌ No data returned")
        subject_clean = subject_dir.rstrip('/')
        if subject_clean.startswith('waves/'):
            subject_clean = subject_clean[6:]
        print(f"URL tried: https://physionet.org/files/mimic4wdb/0.1.0/waves/{subject_clean}/{record_id}/{record_id}n.csv.gz")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and compare MIMIC/UCIQ numerics')
    parser.add_argument('--mode', choices=['diagnose_uciq', 'diagnose_mimic', 'extract_mimic', 
                                           'extract_uciq', 'extract_all'],
                       default='extract_all', help='What to do')
    parser.add_argument('--uciq_file', type=str, help='Single .vital file for diagnosis')
    parser.add_argument('--uciq_dir', type=str, help='Directory with .vital files')
    parser.add_argument('--output_dir', type=str, default='./phase_outputs')
    parser.add_argument('--max_files', type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.mode == 'diagnose_uciq':
        if not args.uciq_file:
            print("ERROR: --uciq_file required for diagnose_uciq mode")
            sys.exit(1)
        diagnose_uciq_file(args.uciq_file)
    
    elif args.mode == 'diagnose_mimic':
        # Test with first record
        records = get_mimic_record_list()
        if records:
            diagnose_mimic_record(records[0]['record_id'], records[0]['subject_dir'])
    
    elif args.mode == 'extract_mimic':
        extract_all_mimic_numerics(output_dir=args.output_dir)
    
    elif args.mode == 'extract_uciq':
        if not args.uciq_dir:
            print("ERROR: --uciq_dir required for extract_uciq mode")
            sys.exit(1)
        extract_all_uciq_numerics(args.uciq_dir, output_dir=args.output_dir, max_files=args.max_files)
    
    elif args.mode == 'extract_all':
        print("STEP 1: MIMIC numerics")
        mimic = extract_all_mimic_numerics(output_dir=args.output_dir)
        
        if args.uciq_dir:
            print("\nSTEP 2: UCIQ numerics")
            uciq = extract_all_uciq_numerics(args.uciq_dir, output_dir=args.output_dir, 
                                             max_files=args.max_files)
        else:
            print("\nSTEP 2: Skipping UCIQ (no --uciq_dir provided)")
        
        print("\nDone! Run extract_numerics_and_compare.py for Phase 2B/4B/6B comparisons.")
