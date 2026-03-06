import pandas as pd
import numpy as np
import os

# --- Config ---
OUTPUT_DIR = "results_auditory"
INPUT_FILE = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "clinical_descriptive_report.md")

def analyze_adoption(df):
    """Analyzes when boxes started and stopped using the system."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    adoption = df.groupby('box').agg(
        first_record=('date', 'min'),
        last_record=('date', 'max'),
        total_hours=('duration_min', lambda x: x.sum() / 60),
        active_days=('date', 'nunique')
    ).reset_index()
    adoption['days_span'] = (adoption['last_record'] - adoption['first_record']).dt.days
    adoption['consistency'] = adoption['active_days'] / (adoption['days_span'] + 1)
    return adoption

def analyze_fragments(df):
    """Analyzes fragment sizes to identify stability vs gaps."""
    # Define thresholds
    # Micro: < 5 min
    # Stable: 5-60 min
    # Long: > 60 min
    df['fragment_type'] = pd.cut(df['duration_min'], 
        bins=[0, 5, 60, 1440, np.inf], 
        labels=['Micro (<5m)', 'Standard (5-60m)', 'Long (>1h)', 'Very Long'])
    frag_dist = df.groupby(['box', 'fragment_type'], observed=False).size().unstack(fill_value=0)
    return frag_dist

def analyze_consistency(df):
    """
    Checks if consecutive recordings in the same box/date 
    have the same monitoring setup (track lists).
    """
    df = df.sort_values(['box', 'date', 'filename'])
    df['prev_tracks'] = df.groupby(['box', 'date'])['tracks'].shift(1)
    df['is_consistent'] = (df['tracks'] == df['prev_tracks']).astype(int)
    
    # Calculate % consistency for sessions with > 1 fragment
    consistency_stats = df.groupby('box')['is_consistent'].mean() * 100
    return consistency_stats

def generate_report(adoption, frags, consistency):
    with open(OUTPUT_REPORT, "w", encoding='utf-8') as f:
        f.write("# Clinical Descriptive Report: ICU Digital Maturity\n\n")
        
        def write_table(title, df, show_index=False):
            f.write(f"## {title}\n")
            try:
                f.write(df.to_markdown(index=show_index) + "\n\n")
            except ImportError:
                f.write("*(Note: Install 'tabulate' for better formatting: pip install tabulate)*\n\n")
                f.write(df.to_string(index=show_index) + "\n\n")

        write_table("1. Digital Adoption & Volume", adoption)
        write_table("2. Fragment Size & Stability", frags, show_index=True)
        write_table("3. Re-connection Consistency", consistency.to_frame("Consistency (%)"), show_index=True)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_clinical_metadata.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    print("Analyzing adoption curves...")
    adoption = analyze_adoption(df)
    
    print("Analyzing fragment distributions...")
    frags = analyze_fragments(df)
    
    print("Analyzing re-connection consistency...")
    consistency = analyze_consistency(df)
    
    print(f"Generating report: {OUTPUT_REPORT}...")
    generate_report(adoption, frags, consistency)
    print("Done!")

if __name__ == "__main__":
    main()
