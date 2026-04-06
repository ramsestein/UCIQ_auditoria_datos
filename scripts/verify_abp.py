#!/usr/bin/env python3
"""Verify MIMIC ABP records for empty headers vs valid data"""

from pathlib import Path
import wfdb
import numpy as np

mimic_dir = Path('data/mimic4wdb_full/waves')

# Get all record directories
record_dirs = [d for d in mimic_dir.iterdir() if d.is_dir()]
print(f'Found {len(record_dirs)} MIMIC record directories')

abp_records = []

for rec_dir in record_dirs[:200]:
    try:
        # Look for header files
        header_files = list(rec_dir.glob('*.hea'))
        if not header_files:
            continue
            
        main_header = header_files[0]
        record_name = main_header.stem
        
        # Read header
        with open(main_header, 'r') as f:
            header_content = f.read()
            
        # Check for ABP in header
        if 'abp' in header_content.lower() or 'art' in header_content.lower():
            # Try to load actual data
            try:
                record = wfdb.rdrecord(str(rec_dir / record_name), physical=False, return_res=64)
                
                # Find ABP channel
                abp_idx = None
                for i, name in enumerate(record.sig_name):
                    if name and ('abp' in name.lower() or 'art' in name.lower()):
                        abp_idx = i
                        break
                
                if abp_idx is not None:
                    abp_signal = record.p_signal[:, abp_idx] if record.p_signal is not None else None
                    
                    if abp_signal is not None:
                        valid_ratio = np.sum(~np.isnan(abp_signal)) / len(abp_signal)
                        valid_values = np.sum((abp_signal > 20) & (abp_signal < 300)) / len(abp_signal)
                        
                        abp_records.append({
                            'record': record_name,
                            'valid_ratio': valid_ratio,
                            'valid_values': valid_values,
                            'has_data': valid_ratio > 0.5 and valid_values > 0.5
                        })
            except Exception as e:
                pass
                
    except Exception as e:
        continue

print(f'\nFound {len(abp_records)} records with ABP channels')
valid_count = sum([r['has_data'] for r in abp_records])
print(f'With valid data (>50% non-NaN): {valid_count}')
print(f'Potentially empty headers: {len(abp_records) - valid_count}')

print('\nABP records details (first 20):')
for r in abp_records[:20]:
    status = 'VALID' if r['has_data'] else 'EMPTY?'
    print(f"  {r['record']}: {r['valid_ratio']:.1%} valid, {r['valid_values']:.1%} reasonable - {status}")
