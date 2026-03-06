import os
import pandas as pd
from pathlib import Path
import importlib.util
import sys

# Import visualize_clinical_audit.py from repo root
spec = importlib.util.spec_from_file_location("visualize_clinical_audit", str(Path(__file__).parents[1] / 'visualize_clinical_audit.py'))
vca = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vca
spec.loader.exec_module(vca)

BASE = Path(__file__).parents[1]
CSV = BASE / 'results_auditory' / 'clinical_metadata_audit.csv'
STABILITY = BASE / 'results_auditory' / 'physiological_complexity_stats.csv'
OUTDIR = BASE / 'results_auditory'

# Threshold for continuous operation (fraction of months active)
# Cambiado a 50% según solicitud
THRESHOLD = 0.5


def identify_continuous_boxes(df, threshold=THRESHOLD):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['month'] = df['date'].dt.to_period('M')

    months_total = df['month'].nunique()
    box_months = df.groupby('box')['month'].nunique()

    # Boxes with months active >= threshold * months_total
    required = max(1, int(months_total * threshold))
    continuous_boxes = box_months[box_months >= required].index.tolist()
    return continuous_boxes, months_total, box_months


def main():
    if not CSV.exists():
        print('clinical_metadata_audit.csv not found.')
        return

    df = pd.read_csv(CSV)
    sdf = pd.read_csv(STABILITY) if STABILITY.exists() else None

    continuous_boxes, months_total, box_months = identify_continuous_boxes(df)

    print(f"Total months in dataset: {months_total}")
    print(f"Boxes detected: {len(box_months)}")
    print(f"Continuous boxes (>= {THRESHOLD*100:.0f}% months): {len(continuous_boxes)}")
    print(continuous_boxes)

    # Filter df
    filtered_df = df[df['box'].isin(continuous_boxes)].copy()

    # Create output filenames with suffix
    os.makedirs(OUTDIR, exist_ok=True)

    # === Prevalence per box for ALL boxes (not filtered) ===
    cols_all = [c for c in df.columns if c.startswith('has_')]
    if cols_all:
        prevalence_all = df.groupby('box')[cols_all].mean() * 100
        plt = vca.plt
        plt.figure(figsize=(12, max(6, 0.4 * len(prevalence_all.index))))
        vca.sns.heatmap(prevalence_all, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Prevalence (%)'})
        plt.title('Signal Category Prevalence by Box (ALL boxes)')
        plt.xlabel('Signal Category')
        plt.ylabel('Box')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'viz_signal_prevalence_all_boxes.png'))
        plt.close()

    # Call plotting functions from visualize_clinical_audit with filtered dataframe
    # General (aggregated) plots (no breakdown by box)
    # 1) Total hours per day (line)
    filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
    daily_total = filtered_df.groupby('date')['duration_min'].sum().sort_index() / 60
    plt = vca.plt
    plt.figure(figsize=(12, 6))
    daily_total.plot(kind='line', marker='o', color='tab:blue', linewidth=2)
    plt.title('Total Hours Recorded (filtered, aggregated)')
    plt.xlabel('Date')
    plt.ylabel('Hours')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'viz_monitoring_adoption_general.png'))
    plt.close()

    # 2) Monthly total (bar) - reuse logic
    vca.plot_general_recordings_by_month(filtered_df)

    # 3) Cumulative monitoring (aggregated)
    cumulative_total = daily_total.cumsum()
    plt.figure(figsize=(12, 6))
    cumulative_total.plot(kind='line', marker='o', color='forestgreen', linewidth=2)
    plt.fill_between(cumulative_total.index, cumulative_total.values, color='forestgreen', alpha=0.2)
    plt.title('Cumulative Clinical Data Growth (filtered, aggregated)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Hours')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'viz_cumulative_monitoring_general.png'))
    plt.close()

    # 4) Signal prevalence overall (bar) instead of heatmap by box
    cols = [c for c in filtered_df.columns if c.startswith('has_')]
    if cols:
        prevalence = filtered_df[cols].mean() * 100
        plt.figure(figsize=(10, 6))
        prevalence.sort_values(ascending=False).plot(kind='bar', color='skyblue', alpha=0.9)
        plt.ylabel('Prevalence (%)')
        plt.title('Signal Category Prevalence (filtered, aggregated)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'viz_signal_prevalence_general.png'))
        plt.close()

    # 5) Complexity vs HR volatility without hue by box
    if sdf is not None:
        merged = pd.merge(filtered_df[['filename', 'complexity_score']],
                          sdf[['filename', 'HR_volatility', 'MAP_volatility']],
                          on='filename')
        if not merged.empty:
            plt.figure(figsize=(10, 6))
            vca.sns.scatterplot(data=merged, x='complexity_score', y='HR_volatility', size='MAP_volatility', alpha=0.6)
            plt.title('Complexity Score vs HR Volatility (filtered, aggregated)')
            plt.xlabel('Complexity Score')
            plt.ylabel('HR Volatility (Std Dev)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, 'viz_complexity_vs_volatility_general.png'))
            plt.close()

    # 6) Media de captación mensual (porcentaje medio de horas capturadas por box)
    # Calculamos por box y mes: capture_pct = (sum duration_min) / (days_in_month * 24 * 60) * 100
    filtered_df['month'] = filtered_df['date'].dt.to_period('M')
    box_month = filtered_df.groupby(['box', 'month'])['duration_min'].sum().reset_index()
    if not box_month.empty:
        def month_days(period):
            # period is pandas Period
            return period.asfreq('D').days_in_month

        # Calculate capture percent per box-month
        box_month['days'] = box_month['month'].apply(lambda p: p.days_in_month if hasattr(p, 'days_in_month') else pd.Period(p).days_in_month)
        box_month['capture_pct'] = box_month['duration_min'] / (box_month['days'] * 24 * 60) * 100

        # Mean capture across boxes per month
        mean_capture = box_month.groupby('month')['capture_pct'].mean()
        # Convert PeriodIndex to timestamp for plotting
        mean_capture.index = mean_capture.index.to_timestamp()

        plt.figure(figsize=(12, 6))
        mean_capture.plot(kind='line', marker='o', color='purple', linewidth=2)
        plt.title('Media de captación mensual (boxes continuos)')
        plt.xlabel('Mes')
        plt.ylabel('Media de captación (%)')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'viz_mean_capture_general.png'))
        plt.close()

    print('Visualizaciones generadas (excluyendo boxes con downtime).')

if __name__ == '__main__':
    main()
