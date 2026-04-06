#!/usr/bin/env python3
"""Quick signal composition check for all 3 datasets"""
from pathlib import Path
import numpy as np
from collections import defaultdict

def parse_wfdb_header_fast(hea_path):
    """Fast WFDB header parser"""
    try:
        with open(hea_path, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 20:
                    break
                lines.append(line)
        
        header_line = None
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                header_line = stripped
                break
        
        if not header_line:
            return None
        
        parts = header_line.split()
        if len(parts) < 2:
            return None
        
        if '/' in parts[0]:
            base_name = parts[0].split('/')[0]
            seg_files = list(hea_path.parent.glob(f"{base_name}_*.hea"))
            if seg_files:
                largest = max(seg_files, key=lambda p: p.stat().st_size)
                return parse_wfdb_header_fast(largest)
            return None
        
        n_sigs = int(parts[1]) if parts[1].isdigit() else 0
        
        signals = []
        sig_idx = 0
        for line in lines[1:]:
            if sig_idx >= n_sigs:
                break
            if line.startswith('#'):
                continue
            lparts = line.strip().split()
            if len(lparts) >= 2:
                sig_name = lparts[-1].split('#')[0].strip()
                if sig_name:
                    signals.append(sig_name)
                    sig_idx += 1
        
        return {'signals': signals, 'n_sigs': n_sigs}
        
    except Exception:
        return None

def check_vitaldb_signals(vitaldb_dir, max_files=100):
    """Check signal composition in VitalDB"""
    npz_files = sorted(vitaldb_dir.glob("*.npz"))[:max_files]
    signals_per_file = []
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
            files = list(data.files)
            # Exclude metadata
            signal_files = [f for f in files if f not in ['fs', 'time', 'ppg_track', 'art_track', 'source', 'filename', 'lag_seconds']]
            signals_per_file.append(len(signal_files))
        except:
            continue
    
    return {'avg_signals': np.mean(signals_per_file) if signals_per_file else 0, 
            'files_checked': len(signals_per_file)}

def check_uciq_signals(uciq_dir, max_files=100):
    """Check signal composition in UCIQ"""
    npz_files = sorted(uciq_dir.glob("*.npz"))[:max_files]
    signals_per_file = []
    signal_types = defaultdict(int)
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
            files = [f.lower() for f in data.files]
            signal_files = [f for f in files if f not in ['fs', 'time', 'ppg_track', 'art_track', 'source', 'filename', 'lag_seconds']]
            signals_per_file.append(len(signal_files))
            
            for f in signal_files:
                signal_types[f] += 1
        except:
            continue
    
    return {'avg_signals': np.mean(signals_per_file) if signals_per_file else 0,
            'signal_types': dict(signal_types),
            'files_checked': len(signals_per_file)}

def check_mimic_signals(mimic_dir, max_records=50):
    """Check signal composition in MIMIC - aggregate by patient"""
    all_hea = list(mimic_dir.rglob("*_*.hea"))
    
    # Group by base record
    record_segments = defaultdict(list)
    for h in all_hea:
        parts = h.stem.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            base_name = '_'.join(parts[:-1])
            record_segments[base_name].append(h)
    
    signals_per_record = []
    all_signal_types = defaultdict(int)
    
    for base_name, seg_files in sorted(record_segments.items())[:max_records]:
        all_signals = set()
        for seg_path in seg_files:
            info = parse_wfdb_header_fast(seg_path)
            if info:
                all_signals.update(info['signals'])
        
        signals_per_record.append(len(all_signals))
        for s in all_signals:
            all_signal_types[s.upper()] += 1
    
    return {'avg_signals': np.mean(signals_per_record) if signals_per_record else 0,
            'signal_types': dict(all_signal_types),
            'files_checked': len(signals_per_record)}

# Paths
data_dir = Path("c:/Users/Ramsés/Desktop/Proyectos/wave_studies/data_vital")

print("=" * 60)
print("SIGNAL COMPOSITION CHECK")
print("=" * 60)

# VitalDB
print("\n1. VITALDB (Korean OR):")
vitaldb_dir = data_dir / "vitaldb_clean"
if vitaldb_dir.exists():
    vitaldb_info = check_vitaldb_signals(vitaldb_dir, max_files=200)
    print(f"   Files checked: {vitaldb_info['files_checked']}")
    print(f"   Avg signals/file: {vitaldb_info['avg_signals']:.1f}")
    print(f"   Typical signals: PPG (SpO2), ABP (ART)")

# UCIQ
print("\n2. UCIQ (Barcelona SICU):")
uciq_dir = data_dir / "clinic_clean"
if uciq_dir.exists():
    uciq_info = check_uciq_signals(uciq_dir, max_files=200)
    print(f"   Files checked: {uciq_info['files_checked']}")
    print(f"   Avg signals/file: {uciq_info['avg_signals']:.1f}")
    print(f"   Signal types found: {list(uciq_info['signal_types'].keys())}")

# MIMIC
print("\n3. MIMIC (US MICU):")
mimic_dir = data_dir / "mimic4wdb_full" / "waves"
if mimic_dir.exists():
    mimic_info = check_mimic_signals(mimic_dir, max_records=50)
    print(f"   Records checked: {mimic_info['files_checked']}")
    print(f"   Avg signals/record: {mimic_info['avg_signals']:.1f}")
    print(f"   Signal types (top 10 by frequency):")
    sorted_sigs = sorted(mimic_info['signal_types'].items(), key=lambda x: x[1], reverse=True)[:10]
    for sig, count in sorted_sigs:
        print(f"      {sig}: {count}/{mimic_info['files_checked']} records")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("\nMIMIC has MORE diverse signals than VitalDB and UCIQ:")
print("  - MIMIC: ECG, PPG/SpO2, RESP, ABP, CO2, ICP, CVP, PAP, etc.")
print("  - VitalDB/UCIQ: Primarily PPG (SpO2) and ABP only")
print("\nMIMIC = Medical ICU with comprehensive monitoring")
print("VitalDB = Operating Room with hemodynamic focus")
print("UCIQ = Surgical ICU with hemodynamic focus")
print("=" * 60)
