#!/usr/bin/env python3
"""Search for numeric channels in MIMIC headers"""

from pathlib import Path

base_dir = Path('data/mimic4wdb_full/waves')

# Search all headers for numeric-like signals
found_signals = set()
signal_counts = {}

# Check multiple records
records = [
    'p100/p10014354/81739927',
    'p100/p10019003/87033314',
    'p101/p10100546/83268087',
]

for rec_path in records:
    rec_dir = base_dir / rec_path
    if not rec_dir.exists():
        continue
    
    hea_files = list(rec_dir.glob('*.hea'))
    for hea_file in hea_files:
        with open(hea_file, 'r') as f:
            lines = f.readlines()
        
        # Parse signal lines (lines with .dat)
        for line in lines:
            if '.dat' in line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 9:
                    sig_name = parts[8]
                    found_signals.add(sig_name)
                    signal_counts[sig_name] = signal_counts.get(sig_name, 0) + 1

print("All unique signal names found:")
for sig in sorted(found_signals):
    print(f"  {sig}: {signal_counts[sig]} occurrences")

# Check for any with '1' in fs field (indicating 1 Hz)
print("\nChecking for 1 Hz sampling rate...")
for rec_path in records[:1]:
    rec_dir = base_dir / rec_path
    hea_files = list(rec_dir.glob('*.hea'))
    
    for hea_file in hea_files[:10]:
        with open(hea_file, 'r') as f:
            lines = f.readlines()
        
        # Get first line (record info)
        if lines:
            first = lines[0].strip()
            parts = first.split()
            if len(parts) >= 3:
                name = parts[0]
                fs = parts[2]
                print(f"  {name}: fs={fs}")
