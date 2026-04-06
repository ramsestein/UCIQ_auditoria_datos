#!/usr/bin/env python3
"""Check MIMIC local file structure"""

from pathlib import Path
import wfdb

base_dir = Path('data/mimic4wdb_full/waves')

# Find main record headers (without underscore)
hea_files = [h for h in base_dir.rglob('*.hea') if '_' not in h.stem]
print(f'Found {len(hea_files)} main .hea files')

# Check first few
for hea in hea_files[:5]:
    print(f'\n{hea.relative_to(base_dir)}')
    try:
        rec = wfdb.rdheader(str(hea.parent / hea.stem))
        print(f'  sig_name: {rec.sig_name}')
        print(f'  fs: {rec.fs}')
        print(f'  n_sig: {rec.n_sig}')
        if rec.seg_name:
            print(f'  seg_name[:3]: {rec.seg_name[:3]}')
    except Exception as e:
        print(f'  Error: {e}')
