#!/usr/bin/env python3
"""Check MIMIC segment signals"""

from pathlib import Path
import wfdb

base_dir = Path('data/mimic4wdb_full/waves')

# Check first segment
seg_path = base_dir / 'p100/p10014354/81739927/81739927_0000'
print(f'Checking segment: {seg_path}')

try:
    seg = wfdb.rdheader(str(seg_path))
    print(f'  sig_name: {seg.sig_name}')
    print(f'  fs: {seg.fs}')
    print(f'  n_sig: {seg.n_sig}')
    
    # Try reading some data
    rec = wfdb.rdrecord(str(seg_path), sampto=100)
    if rec.p_signal is not None:
        print(f'  Data shape: {rec.p_signal.shape}')
        print(f'  First row: {rec.p_signal[0]}')
except Exception as e:
    print(f'  Error: {e}')

# Check another segment
print('\nChecking another segment...')
seg_path2 = base_dir / 'p100/p10019003/87033314/87033314_0000'
try:
    seg2 = wfdb.rdheader(str(seg_path2))
    print(f'  sig_name: {seg2.sig_name}')
except Exception as e:
    print(f'  Error: {e}')
