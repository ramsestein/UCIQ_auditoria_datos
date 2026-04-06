#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Check the extraction results
df = pd.read_pickle('results/mimic_vs_uciq/mimic_numerics_physionet.pkl')

print('Columns:', df.columns.tolist())
print('\nFirst record:')
row = df.iloc[0]
print('Record ID:', row['record_id'])

print('\nValues available:')
for col in df.columns:
    if 'values' in col:
        val = row[col]
        length = len(val) if hasattr(val, '__len__') else 'N/A'
        print(f'  {col}: {length}')
        if length > 0 and isinstance(val, np.ndarray):
            print(f'    Sample values: {val[:5]}')
            break
