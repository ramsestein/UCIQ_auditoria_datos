#!/usr/bin/env python3
"""Check all segment headers for numeric channels"""

from pathlib import Path

base_dir = Path('data/mimic4wdb_full/waves')

# Check all segments in one record
record_dir = base_dir / 'p100/p10014354/81739927'
hea_files = sorted(record_dir.glob('*_*.hea'))

print(f'Found {len(hea_files)} segment headers')
print('\nChecking for numeric-like channels...')

for hea_file in hea_files[:10]:  # Check first 10
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    
    # Get record name and signal count from first line
    first_line = lines[0].strip() if lines else ''
    parts = first_line.split()
    if len(parts) >= 3:
        name = parts[0]
        n_sig = parts[1]
        fs = parts[2] if len(parts) > 2 else '?'
        
        # Check signal lines for numeric indicators
        signal_lines = [l for l in lines if '.dat' in l and not l.startswith('#')]
        has_numeric_fs = any('1 ' in l.split()[2] if len(l.split()) > 2 else False for l in signal_lines)
        
        if has_numeric_fs or int(n_sig) > 0:
            print(f'\n{name}: {n_sig} signals, fs={fs}')
            for line in signal_lines[:5]:  # Show first 5 signals
                parts = line.split()
                if len(parts) >= 9:
                    sig_name = parts[8] if len(parts) > 8 else 'unknown'
                    print(f'  - {sig_name}')
