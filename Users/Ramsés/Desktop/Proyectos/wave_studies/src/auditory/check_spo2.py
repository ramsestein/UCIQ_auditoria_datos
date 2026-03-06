import pandas as pd
from pathlib import Path
p = Path(__file__).parents[1] / 'results_auditory' / 'clinical_metadata_audit.csv'
df = pd.read_csv(p)
total = len(df)
spo2_count = int(df['has_spo2'].fillna(0).astype(int).sum())
print(total, spo2_count)
print('\nPrimeros 20 archivos con has_spo2==1:')
print(df[df['has_spo2']==1]['filename'].head(20).to_list())
print('\nPrimeros 20 archivos con has_spo2==0:')
print(df[df['has_spo2']==0]['filename'].head(20).to_list())
