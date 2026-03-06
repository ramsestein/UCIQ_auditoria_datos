import pandas as pd
import numpy as np
import vitaldb
import os

# --- Config ---
OUTPUT_DIR = "results_auditory"
INPUT_FILE = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")
OUTPUT_PHYSIO = os.path.join(OUTPUT_DIR, "physiological_complexity_stats.csv")

def analyze_physiological_dynamics(filepath):
    """
    Loads raw data for key signals and calculates volatility and 
    % time in physiological range.
    """
    try:
        vf = vitaldb.VitalFile(filepath)
        # Target labels for stability analysis - adjusted for clinic/Intellivue names
        targets = {
            'HR': ['ECG/HR', 'Intellivue/HR', 'Intellivue/ECG_HR'],
            'MAP': ['ART/MBP', 'Intellivue/ABP_M', 'Intellivue/ART_MEAN', 'Intellivue/ABP_MEAN'],
            'SpO2': ['PLETH/SPO2', 'Intellivue/SPO2', 'Intellivue/PLETH_SAT_O2']
        }
        
        results = {}
        
        for label, tracks in targets.items():
            avail = [t for t in vf.get_track_names() if any(x in t for x in tracks)]
            if avail:
                # Load 1 sample every 10 seconds for efficiency
                df = vf.to_pandas(track_names=avail[0], interval=10)
                col = avail[0]
                data = df[col].dropna()
                
                if not data.empty:
                    # Volatility (Standard Deviation)
                    results[f"{label}_volatility"] = data.std()
                    
                    # Time in Range (%)
                    if label == 'HR':
                        in_range = ((data >= 60) & (data <= 100)).mean() * 100
                    elif label == 'MAP':
                        in_range = (data >= 65).mean() * 100
                    elif label == 'SpO2':
                        in_range = (data >= 92).mean() * 100
                    
                    results[f"{label}_in_range_pct"] = in_range
        
        return results
    except Exception as e:
        print(f"Error in physio analysis for {filepath}: {e}")
        return {}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_clinical_metadata.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # Analyze a subset or all depending on size
    # For now, we'll process all files that have at least one of the tracks
    print("Starting physiological stability and volatility analysis...")
    print("Note: This reads raw data and may take time depending on file count.")
    
    physio_results = []
    
    # Process only files > 0 mins for debugging (was > 5)
    target_files = df.copy()
    
    for i, row in target_files.iterrows():
        print(f"[{i+1}/{len(target_files)}] Analyzing dynamics: {row['filename']}")
        dynamics = analyze_physiological_dynamics(row['full_path'])
        if dynamics:
            dynamics['filename'] = row['filename']
            dynamics['box'] = row['box']
            physio_results.append(dynamics)
            
    if physio_results:
        pdf = pd.DataFrame(physio_results)
        pdf.to_csv(OUTPUT_PHYSIO, index=False)
        
        # Summary by box
        summary = pdf.groupby('box').mean().reset_index()
        print("\nPhysiological Stability Summary (Mean % in range / SD):")
        print(summary.to_string())
        print(f"\nSaved detailed dynamics to {OUTPUT_PHYSIO}")
    else:
        print("No stability metrics could be extracted.")

if __name__ == "__main__":
    main()
