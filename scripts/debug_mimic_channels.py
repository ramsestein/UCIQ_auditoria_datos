#!/usr/bin/env python3
"""Debug: Check actual channel names in MIMIC records"""

import wfdb

# Try using the mimic4wdb database directly
print(f"{'='*60}")
print("Checking MIMIC4WDB 0.1.0 structure")
print('='*60)

# Try to get record list
print("\nTrying to get record list...")
try:
    records = wfdb.io.record.get_record_list('mimic4wdb/0.1.0')
    print(f"Found {len(records)} records")
    print(f"First 5: {records[:5]}")
except Exception as e:
    print(f"Error: {e}")

# Try with waves subdirectory
print("\nTrying mimic4wdb/0.1.0/waves...")
try:
    records = wfdb.io.record.get_record_list('mimic4wdb/0.1.0/waves')
    print(f"Found {len(records)} records")
    print(f"First 5: {records[:5]}")
except Exception as e:
    print(f"Error: {e}")

# Try accessing a specific record with full db path
print("\nTrying to read with full database path...")
try:
    rec = wfdb.rdrecord('p100/p10014354', pn_dir='mimic4wdb/0.1.0/waves/p100')
    print(f"Success!")
    print(f"  Signals: {rec.sig_name}")
except Exception as e:
    print(f"Error: {e}")
