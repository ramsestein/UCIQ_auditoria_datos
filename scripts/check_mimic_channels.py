#!/usr/bin/env python3
"""Debug MIMIC channel names"""

import wfdb
from pathlib import Path

# Check a sample record
rec_dir = Path('data/mimic4wdb_full/waves/p100')
header_files = list(rec_dir.glob('*.hea'))
print('Header files:', [h.name for h in header_files[:5]])

if header_files:
    rec_name = header_files[0].stem
    print(f'Record name: {rec_name}')
    
    # Read header
    rec = wfdb.rdheader(str(rec_dir / rec_name))
    seg_info = rec.seg_name[:5] if rec.seg_name else None
    print(f'Segments: {seg_info}')
    
    # Read first segment if multi-segment
    if rec.seg_name and rec.seg_name[0]:
        seg_path = str(rec_dir / rec.seg_name[0])
        print(f'Reading segment: {rec.seg_name[0]}')
        try:
            seg = wfdb.rdrecord(seg_path)
            print(f'Segment signals: {seg.sig_name}')
            shape = seg.p_signal.shape if seg.p_signal is not None else None
            print(f'Signal shape: {shape}')
        except Exception as e:
            print(f'Error reading segment: {e}')
