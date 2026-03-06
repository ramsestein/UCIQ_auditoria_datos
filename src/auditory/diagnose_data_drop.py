import pandas as pd
import os

INPUT_FILE = r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies\results_auditory\clinical_metadata_audit.csv"

if not os.path.exists(INPUT_FILE):
    print("Error: metadata file not found.")
else:
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Analysis 1: General Monthly Volume
    general_stats = df.groupby('month').agg(
        total_hours=('duration_min', lambda x: x.sum() / 60),
        file_count=('filename', 'count'),
        active_boxes=('box', 'nunique')
    )
    
    print("--- General Monthly Statistics ---")
    print(general_stats)
    
    # Analysis 2: When each box stopped monitoring
    box_stops = df.groupby('box')['date'].max().sort_values()
    print("\n--- Last Recording Date per Box ---")
    print(box_stops)
    
    # Analysis 3: Heatmap-style breakdown (Hours per Month per Box)
    cross_tab = df.pivot_table(index='month', columns='box', values='duration_min', aggfunc='sum', fill_value=0) / 60
    print("\n--- Hours per box per month (Sample) ---")
    print(cross_tab.tail(6))
