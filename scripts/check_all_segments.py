#!/usr/bin/env python3
"""Check all segments in a MIMIC record for numeric channels"""

from pathlib import Path
import wfdb

base_dir = Path('data/mimic4wdb_full/waves')

# Check all segments in first record
record_dir = base_dir / 'p100/p10014354/81739927'
seg_files = sorted(record_dir.glob('*.hea'))

print(f'Found {len(seg_files)} segments in {record_dir}')
print('\nSegment signals:')

for seg_file in seg_files:
    seg_name = seg_file.stem
    try:
        seg = wfdb.rdheader(str(record_dir / seg_name))
        signals = seg.sig_name if seg.sig_name else ['(layout)']
        print(f'  {seg_name}: {signals}')
    except Exception as e:
        print(f'  {seg_name}: Error - {e}')
