"""
generate_table1.py
──────────────────────────────────────────────────────────────────────
Genera la Tabla 1 descriptiva formal para el artículo científico.
Incluye: características del dataset, disponibilidad de señales,
tasas de muestreo, calidad, artefactos y duración.

Salida:
  - table1_descriptive.csv       (tabla completa con todas las métricas)
  - table1_descriptive.md        (versión Markdown lista para artículo)
  - table1_signal_detail.csv     (tabla detallada de señales)
──────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
RESULTS = BASE / "results_auditory"
RESULTS_FINAL = RESULTS / "results_final"
OUT = RESULTS


def q1q3(series):
    """Return Q1-Q3 string."""
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}"


def mean_sd(series, fmt=".1f"):
    """Return mean ± SD string."""
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{s.mean():{fmt}} ± {s.std():{fmt}}"


def median_iqr(series, fmt=".1f"):
    """Return median [IQR] string."""
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    return f"{s.median():{fmt}} [{s.quantile(0.25):{fmt}}–{s.quantile(0.75):{fmt}}]"


def main():
    print("=" * 60)
    print("GENERATING TABLE 1 — Descriptive Statistics")
    print("=" * 60)

    # ─── Load all data sources ───────────────────────────────────────
    meta = pd.read_csv(RESULTS / "clinical_metadata_audit.csv")
    quality = pd.read_csv(RESULTS / "quality_sample_summary.csv")
    sampling = pd.read_csv(RESULTS / "sampling_rates_summary.csv")
    duration = pd.read_csv(RESULTS / "session_duration_stats.csv")
    artifacts = pd.read_csv(RESULTS_FINAL / "artifact_detection_summary.csv")
    completeness = pd.read_csv(RESULTS_FINAL / "completeness_by_box.csv")
    gaps = pd.read_csv(RESULTS_FINAL / "gap_stats_overall.csv")

    # ─── Panel A: Dataset Characteristics ────────────────────────────
    n_files = len(meta)
    n_unique = meta["filename"].nunique()
    n_boxes = meta["box"].nunique()
    box_list = sorted(meta["box"].dropna().unique())
    total_hours = meta["duration_min"].sum() / 60
    dur = meta["duration_min"].dropna()
    date_range_start = meta["date"].dropna().min()
    date_range_end = meta["date"].dropna().max()

    # Duration categories from session_duration_stats
    dur_stats = duration.iloc[0] if len(duration) > 0 else {}

    rows = []

    # Panel A header
    rows.append({"Category": "Dataset Characteristics", "Variable": "", "Value": ""})
    rows.append({"Category": "", "Variable": "Collection period", "Value": f"{date_range_start} to {date_range_end}"})
    rows.append({"Category": "", "Variable": "Total monitoring files, n", "Value": f"{n_files:,}"})
    rows.append({"Category": "", "Variable": "Unique filenames, n", "Value": f"{n_unique:,}"})
    rows.append({"Category": "", "Variable": "Bedside monitors (boxes), n", "Value": str(n_boxes)})
    rows.append({"Category": "", "Variable": "Total monitoring hours", "Value": f"{total_hours:,.1f}"})
    rows.append({"Category": "", "Variable": "Session duration, mean ± SD (min)", "Value": mean_sd(dur)})
    rows.append({"Category": "", "Variable": "Session duration, median [IQR] (min)", "Value": median_iqr(dur)})
    rows.append({"Category": "", "Variable": "Session duration, range (min)", "Value": f"{dur.min():.1f}–{dur.max():.1f}"})
    rows.append({"Category": "", "Variable": "Micro sessions (<5 min), %", "Value": f"{(dur < 5).mean() * 100:.1f}"})
    rows.append({"Category": "", "Variable": "Short sessions (5–30 min), %", "Value": f"{((dur >= 5) & (dur < 30)).mean() * 100:.1f}"})
    rows.append({"Category": "", "Variable": "Standard sessions (30–60 min), %", "Value": f"{((dur >= 30) & (dur < 60)).mean() * 100:.1f}"})
    rows.append({"Category": "", "Variable": "Long sessions (1–4 h), %", "Value": f"{((dur >= 60) & (dur < 240)).mean() * 100:.1f}"})
    rows.append({"Category": "", "Variable": "Extended sessions (>4 h), %", "Value": f"{(dur >= 240).mean() * 100:.1f}"})
    rows.append({"Category": "", "Variable": "Tracks per file, median [IQR]", "Value": median_iqr(meta["track_count"], fmt=".0f")})

    # Panel B: Signal Availability
    rows.append({"Category": "Signal Availability", "Variable": "", "Value": ""})
    sig_info = [
        ("ECG (any lead)", "has_ecg"),
        ("SpO2 / Plethysmography", "has_spo2"),
        ("Arterial blood pressure", "has_art"),
        ("Capnography (CO2)", "has_co2"),
        ("Hemodynamics (CVP/PAP/CO)", "has_hemo"),
        ("Neurological (ICP/EEG)", "has_neuro"),
        ("Ventilation parameters", "has_vent"),
        ("BIS / Depth of anesthesia", "has_bis"),
        ("Temperature", "has_temp"),
    ]
    for label, col in sig_info:
        if col in meta.columns:
            pct = meta[col].mean() * 100
            n = int(meta[col].sum())
            rows.append({"Category": "", "Variable": f"{label}, n (%)", "Value": f"{n:,} ({pct:.1f}%)"})

    # Complexity score
    rows.append({"Category": "", "Variable": "Complexity score, median [IQR]", "Value": median_iqr(meta["complexity_score"], fmt=".0f")})

    # Panel C: Sampling Rates
    rows.append({"Category": "Sampling Rates", "Variable": "", "Value": ""})
    key_tracks = [
        ("ECG waveform (Lead II)", "INTELLIVUE/ECG_II"),
        ("ECG waveform (Lead III)", "INTELLIVUE/ECG_III"),
        ("Plethysmography waveform", "INTELLIVUE/PLETH"),
        ("Respiration waveform", "INTELLIVUE/RESP"),
        ("Numeric trends (HR, SpO2, etc.)", None),  # Generic
    ]
    
    # Check column name for Hz
    hz_col = "median_hz" if "median_hz" in sampling.columns else "median_srate"
    
    for label, track in key_tracks:
        if track and track.upper() in sampling["track"].str.upper().values:
            mask = sampling["track"].str.upper() == track.upper()
            row_data = sampling[mask].iloc[0]
            hz = row_data[hz_col]
            rows.append({"Category": "", "Variable": f"{label} (Hz)", "Value": f"{hz:.1f}"})
        elif track is None:
            rows.append({"Category": "", "Variable": f"{label} (Hz)", "Value": "1.0"})

    rows.append({"Category": "", "Variable": "Unique track types, n", "Value": str(len(sampling))})
    n_waveform = len(sampling[sampling[hz_col] > 10]) if hz_col in sampling.columns else "N/A"
    n_numeric = len(sampling[sampling[hz_col] <= 10]) if hz_col in sampling.columns else "N/A"
    rows.append({"Category": "", "Variable": "Waveform tracks (>10 Hz), n", "Value": str(n_waveform)})
    rows.append({"Category": "", "Variable": "Numeric trend tracks (≤10 Hz), n", "Value": str(n_numeric)})

    # Panel D: Data Quality
    rows.append({"Category": "Data Quality", "Variable": "", "Value": ""})
    q_non_nan = quality["pct_non_nan"]
    q_std = quality["std"].dropna()
    q_flat = quality["flatline_pct"]

    rows.append({"Category": "", "Variable": "Files with quality assessment, n", "Value": f"{len(quality):,}"})
    rows.append({"Category": "", "Variable": "Signal completeness (% non-NaN), mean ± SD", "Value": mean_sd(q_non_nan)})
    rows.append({"Category": "", "Variable": "Signal completeness, median [IQR]", "Value": median_iqr(q_non_nan)})
    rows.append({"Category": "", "Variable": "Signal variability (std), mean ± SD", "Value": mean_sd(q_std)})
    rows.append({"Category": "", "Variable": "Flatline percentage, mean ± SD", "Value": mean_sd(q_flat)})
    rows.append({"Category": "", "Variable": "Files with >90% completeness, %", "Value": f"{(q_non_nan > 90).mean() * 100:.1f}"})
    rows.append({"Category": "", "Variable": "Files with >0% flatline, %", "Value": f"{(q_flat > 0).mean() * 100:.1f}"})

    # Panel E: Artifact Detection
    rows.append({"Category": "Artifact Detection", "Variable": "", "Value": ""})
    rows.append({"Category": "", "Variable": "Signals analyzed, n", "Value": str(len(artifacts))})
    
    key_artifact_signals = ["ECG_HR", "PLETH_SAT_O2", "RR", "ABP_MEAN", "ABP_SYS", "CO2_ET", "TEMP", "ICP"]
    for sig in key_artifact_signals:
        mask = artifacts["signal"] == sig
        if mask.any():
            art_row = artifacts[mask].iloc[0]
            pct = art_row.get("overall_pct_artifact", art_row.get("mean_pct_artifact", 0))
            desc = art_row.get("signal_desc", sig)
            rows.append({"Category": "", "Variable": f"{desc} artifact rate (%)", "Value": f"{pct:.3f}"})

    # Panel F: Temporal Gaps
    rows.append({"Category": "Temporal Gaps", "Variable": "", "Value": ""})
    if len(gaps) > 0:
        # gaps is stored transposed — extract values
        gap_vals = {}
        for _, row in gaps.iterrows():
            gap_vals[row.iloc[0]] = row.iloc[1]
        rows.append({"Category": "", "Variable": "Inter-session gaps, n", "Value": f"{int(float(gap_vals.get('count', 0))):,}"})
        rows.append({"Category": "", "Variable": "Gap duration, median (h)", "Value": f"{float(gap_vals.get('median_hours', 0)):.1f}"})
        rows.append({"Category": "", "Variable": "Gap duration, mean (h)", "Value": f"{float(gap_vals.get('mean_hours', 0)):.1f}"})
        rows.append({"Category": "", "Variable": "Gap duration, P90 (h)", "Value": f"{float(gap_vals.get('p90_hours', 0)):.1f}"})

    # Panel G: Per-Box Summary
    rows.append({"Category": "Per-Box Summary", "Variable": "", "Value": ""})
    box_counts = meta.groupby("box").size()
    box_hours = meta.groupby("box")["duration_min"].sum() / 60
    rows.append({"Category": "", "Variable": "Files per box, median [IQR]", "Value": median_iqr(box_counts, fmt=".0f")})
    rows.append({"Category": "", "Variable": "Hours per box, median [IQR]", "Value": median_iqr(box_hours, fmt=".0f")})
    rows.append({"Category": "", "Variable": "Most active box (files)", "Value": f"{box_counts.idxmax()} (n={box_counts.max():,})"})
    rows.append({"Category": "", "Variable": "Least active box (files)", "Value": f"{box_counts.idxmin()} (n={box_counts.min():,})"})

    # ─── Create DataFrame and save ──────────────────────────────────
    df_table1 = pd.DataFrame(rows)
    df_table1.to_csv(OUT / "table1_descriptive.csv", index=False)
    print(f"\n  -> Saved table1_descriptive.csv ({len(df_table1)} rows)")

    # ─── Generate Markdown version ───────────────────────────────────
    md_lines = [
        "# Table 1. Descriptive Statistics of the Clinical Monitoring Dataset",
        "",
        "| Category | Variable | Value |",
        "|----------|----------|-------|",
    ]
    for _, row in df_table1.iterrows():
        cat = row["Category"]
        var = row["Variable"]
        val = row["Value"]
        if cat and not var:
            md_lines.append(f"| **{cat}** | | |")
        else:
            md_lines.append(f"| | {var} | {val} |")
    
    md_lines.extend([
        "",
        f"*Data collected from {n_boxes} GE CARESCAPE bedside monitors across {date_range_start} to {date_range_end}.*",
        f"*Total dataset: {n_files:,} files, {total_hours:,.1f} monitoring hours.*",
        f"*Quality assessed on {len(quality):,} files. Artifact detection on {len(artifacts)} signal types.*",
    ])

    md_text = "\n".join(md_lines)
    (OUT / "table1_descriptive.md").write_text(md_text, encoding="utf-8")
    print(f"  -> Saved table1_descriptive.md")

    # ─── Signal Detail Table ─────────────────────────────────────────
    sig_detail = []
    for label, col in sig_info:
        if col not in meta.columns:
            continue
        pct = meta[col].mean() * 100
        n = int(meta[col].sum())

        # Find artifact rate for this signal category
        art_rate = "N/A"
        artifact_map = {
            "has_ecg": "ECG_HR", "has_spo2": "PLETH_SAT_O2", "has_art": "ABP_MEAN",
            "has_co2": "CO2_ET", "has_temp": "TEMP", "has_bis": "BIS",
        }
        art_sig = artifact_map.get(col)
        if art_sig:
            mask = artifacts["signal"] == art_sig
            if mask.any():
                art_rate = f"{artifacts[mask].iloc[0]['overall_pct_artifact']:.3f}%"

        # Find completeness variation by box
        comp_col_map = {
            "has_ecg": "has_ecg", "has_spo2": "has_spo2", "has_art": "has_art",
            "has_co2": "has_co2", "has_bis": "has_bis", "has_neuro": "has_neuro",
            "has_hemo": "has_hemo", "has_vent": "has_vent", "has_temp": "has_temp",
        }
        comp_col = comp_col_map.get(col)
        if comp_col and comp_col in completeness.columns:
            box_range = f"{completeness[comp_col].min():.1f}–{completeness[comp_col].max():.1f}%"
        else:
            box_range = "N/A"

        sig_detail.append({
            "Signal Category": label,
            "N files": n,
            "Overall %": f"{pct:.1f}",
            "Box range %": box_range,
            "Artifact rate": art_rate,
        })

    df_sig = pd.DataFrame(sig_detail)
    df_sig.to_csv(OUT / "table1_signal_detail.csv", index=False)
    print(f"  -> Saved table1_signal_detail.csv ({len(df_sig)} rows)")

    # ─── Print Table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 1 — DESCRIPTIVE STATISTICS")
    print("=" * 70)
    for _, row in df_table1.iterrows():
        cat = row["Category"]
        var = row["Variable"]
        val = row["Value"]
        if cat and not var:
            print(f"\n  [{cat}]")
        elif var:
            print(f"    {var:<55s} {val}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
