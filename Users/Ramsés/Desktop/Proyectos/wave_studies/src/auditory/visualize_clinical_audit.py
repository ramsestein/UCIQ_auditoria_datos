import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Config ---
OUTPUT_DIR = "results_auditory"
INPUT_FILE = os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv")
STABILITY_FILE = os.path.join(OUTPUT_DIR, "physiological_complexity_stats.csv")

def plot_adoption_timeline(df):
    """Plots total hours recorded per box over time."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    daily_box = df.groupby(['date', 'box'])['duration_min'].sum().unstack(fill_value=0) / 60
    
    plt.figure(figsize=(12, 6))
    daily_box.plot(kind='area', stacked=True, alpha=0.7, ax=plt.gca())
    plt.title("ICU Monitoring Adoption: Hours per Box Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Hours Recorded")
    plt.legend(title="Box", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "viz_monitoring_adoption.png"))
    plt.close()

def plot_signal_prevalence(df):
    """Heatmap of signal categories across boxes."""
    cols = [c for c in df.columns if c.startswith('has_')]
    prevalence = df.groupby('box')[cols].mean() * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(prevalence, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Prevalence (%)'})
    plt.title("Signal Category Prevalence by Box")
    plt.xlabel("Signal Category")
    plt.ylabel("Box")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "viz_signal_prevalence.png"))
    plt.close()

def plot_complexity_vs_stability(df, sdf):
    """Scatter plot: Complexity Score vs Physiological Volatility."""
    if sdf is None or df is None:
        return
        
    # Merge on filename
    merged = pd.merge(df[['filename', 'complexity_score', 'box']], 
                      sdf[['filename', 'HR_volatility', 'MAP_volatility']], 
                      on='filename')
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x='complexity_score', y='HR_volatility', hue='box', size='MAP_volatility', alpha=0.6)
    plt.title("Clinical Complexity vs. HR Volatility")
    plt.xlabel("Complexity Score (Signal Count/Type)")
    plt.ylabel("HR Volatility (Std Dev)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "viz_complexity_vs_volatility.png"))
    plt.close()

def plot_general_recordings_by_month(df):
    """Plots total hours recorded per month (all boxes combined)."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Resample to month and sum
    monthly_total = df.set_index('date')['duration_min'].resample('ME').sum() / 60
    
    plt.figure(figsize=(12, 6))
    monthly_total.plot(kind='bar', color='skyblue', alpha=0.8)
    
    # Format labels to show Year-Month
    plt.gca().set_xticklabels([d.strftime('%Y-%m') for d in monthly_total.index])
    
    plt.title("General ICU Monitoring Volume: Total Hours per Month")
    plt.xlabel("Month")
    plt.ylabel("Total Hours Recorded")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "viz_general_records_by_month.png"))
    plt.close()

def plot_cumulative_monitoring(df):
    """Plots cumulative monitoring hours over time."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    daily_total = df.groupby('date')['duration_min'].sum().sort_index() / 60
    cumulative_total = daily_total.cumsum()
    
    plt.figure(figsize=(12, 6))
    cumulative_total.plot(kind='line', marker='o', color='forestgreen', linewidth=2)
    plt.fill_between(cumulative_total.index, cumulative_total.values, color='forestgreen', alpha=0.2)
    plt.title("Cumulative Clinical Data Growth")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Hours Recorded")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "viz_cumulative_monitoring.png"))
    plt.close()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    sdf = pd.read_csv(STABILITY_FILE) if os.path.exists(STABILITY_FILE) else None
    
    print("Generating visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plot_adoption_timeline(df)
    # Filtrar cajas continuas (>=50% meses activos) sólo para la gráfica general mensual
    try:
        tmp = df.copy()
        tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
        tmp = tmp.dropna(subset=['date'])
        tmp['month'] = tmp['date'].dt.to_period('M')
        months_total = tmp['month'].nunique()
        box_months = tmp.groupby('box')['month'].nunique()
        threshold = 0.5
        required = max(1, int(months_total * threshold))
        continuous_boxes = box_months[box_months >= required].index.tolist()
        filtered_df = df[df['box'].isin(continuous_boxes)].copy()
        print(f"Using continuous boxes for general monthly plot (>=50% months): {continuous_boxes}")
    except Exception:
        filtered_df = df

    plot_general_recordings_by_month(filtered_df)
    plot_cumulative_monitoring(df)
    plot_signal_prevalence(df)
    if sdf is not None:
        plot_complexity_vs_stability(df, sdf)
        
    print("Visualizations saved to results_auditory/")

if __name__ == "__main__":
    main()
