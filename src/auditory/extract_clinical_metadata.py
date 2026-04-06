import os
import pandas as pd
import vitaldb
from datetime import datetime
import json

# --- Config ---
DATA_DIR = r"C:\Users\Ramsés\Desktop\Proyectos\wave_studies\data\clinic"
OUTPUT_DIR = "results_auditory"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_vital_metadata(filepath):
    """Extracts clinical metadata from a single .vital file."""
    try:
        vf = vitaldb.VitalFile(filepath)
        tracks = vf.get_track_names()
        
        # Basic timing
        # vitaldb files usually have dt and tracks. 
        # We need start/end to calculate duration.
        # vf.info might contain some metadata, but usually we look at tracks.
        
        duration_sec = 0
        if tracks:
            # Get common sample rate or just the max duration from any track
            # vf.to_pandas is heavy, let's use vf.get_track_names and internal info if possible
            # Simplified: duration based on track length and interval
            # In clinical .vital files, tracks often have different lengths.
            # We'll approximate for now using the VitalFile object properties if available.
            pass
        
        # In clinical files, sometimes lengths are not in headers.
        # We need to find the max 'time' across tracks.
        # Active sampling: get the first and last valid timestamps if possible.
        # Better duration logic: 
        # In VitalDB, the 'nsamp' and 'dt' in trks info are fast.
        # However, to be 100% sure without loading all data, we sample carefully.
        max_duration = 0
        if tracks:
            try:
                # Get the first track info to estimate duration
                # VitalDB files usually have a consistent time axis if multiple tracks exist.
                # If not, vf.to_pandas is the safest but we use a large interval.
                df_time = vf.to_pandas(track_names=[tracks[0]], interval=10)
                if not df_time.empty:
                    max_duration = len(df_time) * 10
            except Exception as e:
                print(f"    [Debug] Sampling failed for {os.path.basename(filepath)}: {e}")

        # Expanded categories for complexity and clinical maturity audit
        has_ecg = any(any(x in t.upper() for x in ["ECG", "HR"]) for t in tracks)
        has_art = any(any(x in t.upper() for x in ["ART", "ABP", "MBP"]) for t in tracks)
        has_co2 = any(any(x in t.upper() for x in ["CO2", "CAPNO"]) for t in tracks)
        has_spo2 = any("SPO2" in t.upper() for t in tracks)
        has_bis = any("BIS" in t.upper() for t in tracks)
        
        # Neuro: BIS, EEG, ICP
        has_neuro = any(any(x in t.upper() for x in ["EEG", "BIS", "ICP"]) for t in tracks)
        
        # Hemodynamics (Advanced): CVP, PAP, CO, SV, etc.
        hemo_keywords = ["CVP", "PAP", "CO", "SV", "PCWP", "SVR", "PVR"]
        has_hemo = any(any(x in t.upper() for x in hemo_keywords) for t in tracks)
        
        # Ventilation: Flow, Volume, PEEP, AWP
        vent_keywords = ["AWP", "FLOW", "VOL", "PEEP", "VENT", "TV_", "MVE"]
        has_vent = any(any(x in t.upper() for x in vent_keywords) for t in tracks)
        
        # Metabolic/Temp
        has_temp = any("TEMP" in t.upper() for t in tracks)

        stats = {
            "filename": os.path.basename(filepath),
            "full_path": filepath,
            "duration_min": round(max_duration / 60, 2),
            "track_count": len(tracks),
            "has_ecg": int(has_ecg),
            "has_art": int(has_art),
            "has_co2": int(has_co2),
            "has_spo2": int(has_spo2),
            "has_bis": int(has_bis),
            "has_neuro": int(has_neuro),
            "has_hemo": int(has_hemo),
            "has_vent": int(has_vent),
            "has_temp": int(has_temp),
            "complexity_score": int(has_ecg) + int(has_art) + int(has_co2) + int(has_neuro) + int(has_hemo) + int(has_vent) + int(has_temp),
            "tracks": ",".join(tracks)
        }
        return stats
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    results = []
    
    print(f"Scanning {DATA_DIR}...")
    
    # Structure: data/clinic/boxN/PATIENT_ID/DATE/file.vital
    # OR data/clinic/boxN/file.vital (from my exploration)
    
    for root, dirs, files in os.walk(DATA_DIR):
        # Extract Box Info from path
        parts = root.split(os.sep)
        box_name = "unknown"
        for p in parts:
            if p.startswith("box"):
                box_name = p
                break
        
        for f in files:
            if f.endswith(".vital"):
                fpath = os.path.join(root, f)
                print(f"Processing {box_name} - {f}")
                meta = get_vital_metadata(fpath)
                if meta:
                    meta["box"] = box_name
                    # Try to extract date from filename (e.g., fdrbsau47_250723_034554.vital)
                    try:
                        name_parts = f.split("_")
                        if len(name_parts) >= 2:
                            date_str = name_parts[1] # "250723"
                            meta["date"] = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"
                    except:
                        meta["date"] = "unknown"
                    
                    results.append(meta)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAudit complete! Result saved to {OUTPUT_FILE}")
        print(df.groupby("box")["duration_min"].sum().reset_index().rename(columns={"duration_min": "total_min"}))
    else:
        print("No .vital files found or processed.")

if __name__ == "__main__":
    main()
