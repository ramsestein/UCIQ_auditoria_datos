"""
compare_mimic_biosignals.py
──────────────────────────────────────────────────────────────────────
Comparación estructurada entre el dataset clínico propio (VitalDB/.vital)
y los datos públicos MIMIC-IV Waveform + MIMIC-III Numeric Subset.

Genera:
  - mimic_waveform_summary.csv    (señales, fs, duración por registro MIMIC-IV waveform)
  - mimic_numeric_summary.csv     (señales y presencia en MIMIC-III numeric subset)
  - comparison_table.csv          (tabla comparativa formal Clínico vs MIMIC)
  - comparison_radar.png          (radar chart de disponibilidad de señales)
  - comparison_bar_signals.png    (barras de señales comparadas)
──────────────────────────────────────────────────────────────────────
"""

import os, re, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BASE = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
MIMIC_WAVES = BASE / "data" / "mimic4" / "waves"
RESULTS = BASE / "results_auditory"
OUT = RESULTS  # output alongside other results

# MIMIC-IV Waveform DB v0.1.0 real proportions (200 records, 198 patients)
# Source: Moody et al., PhysioNet, DOI: 10.13026/a2mw-f949
MIMIC4_TOTAL_RECORDS = 200
MIMIC4_TOTAL_PATIENTS = 198
MIMIC4_PROPORTIONS_CSV = RESULTS / "mimic_waveform_proportions.csv"


def load_mimic4_proportions():
    """Load the real signal proportions from the full MIMIC-IV Waveform DB."""
    df = pd.read_csv(MIMIC4_PROPORTIONS_CSV)
    # Build a dict: signal_name → proportion (0-1)
    props = {}
    for _, row in df.iterrows():
        props[row["signal"]] = float(row["proportion"])
    return df, props


# ─── 1. PARSE MIMIC-IV WAVEFORM MASTER HEADERS ──────────────────────────────

def parse_master_hea(hea_path):
    """Parse a MIMIC-IV multi-segment master .hea file."""
    info = {
        "record_id": hea_path.stem,
        "patient_dir": str(hea_path.parent.parent.name),
        "path": str(hea_path),
    }
    with open(hea_path, "r") as f:
        lines = f.readlines()

    # Skip comment lines (e.g. "#wfdb 10.7")
    data_lines = [l for l in lines if not l.startswith("#")]
    comment_lines = [l for l in lines if l.startswith("#")]

    # First data line: record_name/n_seg n_sig fs/counter_freq total_samples [start_time] [date]
    header = data_lines[0].strip().split()
    parts = header[0].split("/")
    n_seg = int(parts[1]) if len(parts) > 1 else 1
    info["n_signals"] = int(header[1])
    
    # Parse fs — may be "fs/counter_freq"
    fs_str = header[2]
    info["fs_base"] = float(fs_str.split("/")[0])
    info["total_samples"] = int(header[3])
    info["duration_hours"] = info["total_samples"] / info["fs_base"] / 3600 if info["fs_base"] > 0 else 0

    if len(header) >= 6:
        info["start_time"] = header[4]
        info["date"] = header[5]

    # Count valid segments (not gaps '~') and parse signal names from segment headers
    segment_lines = [l for l in data_lines[1:] if l.strip()]
    valid_segs = [l for l in segment_lines if not l.strip().startswith("~")]
    info["n_segments"] = n_seg
    info["n_valid_segments"] = len(valid_segs)
    gap_segs = [l for l in segment_lines if l.strip().startswith("~")]
    info["n_gaps"] = len(gap_segs)

    # Parse comments for subject/hadm
    for l in comment_lines:
        if "subject_id" in l:
            info["subject_id"] = l.split()[-1]
        elif "hadm_id" in l:
            info["hadm_id"] = l.split()[-1]

    return info


def parse_segment_hea(hea_path):
    """Parse a single segment .hea to extract signal names, units, and effective sampling rates."""
    signals = []
    with open(hea_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split()
    n_sig = int(header[1])
    fs_str = header[2]
    fs_base = float(fs_str.split("/")[0].split("(")[0])
    n_samples = int(header[3])

    for line in lines[1:n_sig + 1]:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        sig_name = parts[-1]
        # Format field (index 1) can reveal multiplexing: e.g. "516x4"
        fmt = parts[1]
        multiplier = 1
        if "x" in fmt:
            multiplier = int(fmt.split("x")[1])
        effective_fs = fs_base * multiplier
        
        unit_field = parts[2]  # "gain(baseline)/unit" or "gain/unit"
        unit = unit_field.split("/")[-1] if "/" in unit_field else ""
        
        signals.append({
            "signal": sig_name,
            "unit": unit,
            "effective_fs": effective_fs,
            "multiplier": multiplier,
        })
    return signals, n_samples, fs_base


def analyze_mimic_waveforms():
    """Scan all downloaded MIMIC-IV waveform records."""
    records = []
    all_signals = defaultdict(lambda: {"count": 0, "fs_values": [], "units": set()})

    # Find master headers (no underscore in stem = master record)
    for hea in MIMIC_WAVES.rglob("*.hea"):
        if "_" in hea.stem:
            continue
        # Check it's a real multi-segment master (not an empty placeholder)
        try:
            info = parse_master_hea(hea)
        except Exception as e:
            print(f"  Skipping {hea}: {e}")
            continue

        if info["total_samples"] == 0:
            continue

        # Parse segment headers for signal inventory
        seg_dir = hea.parent
        seg_signals = set()
        seg_fs = {}
        for seg_hea in sorted(seg_dir.glob(f"{hea.stem}_*.hea")):
            try:
                sigs, _, _ = parse_segment_hea(seg_hea)
                for s in sigs:
                    seg_signals.add(s["signal"])
                    seg_fs[s["signal"]] = s["effective_fs"]
                    all_signals[s["signal"]]["count"] += 1
                    all_signals[s["signal"]]["fs_values"].append(s["effective_fs"])
                    all_signals[s["signal"]]["units"].add(s["unit"])
            except Exception:
                continue

        info["signals"] = ", ".join(sorted(seg_signals))
        info["n_unique_signals"] = len(seg_signals)
        for sig in seg_signals:
            info[f"fs_{sig}"] = seg_fs.get(sig, "")
        records.append(info)

    df_records = pd.DataFrame(records)
    
    # Signal summary
    sig_rows = []
    for sig, data in all_signals.items():
        fs_arr = np.array(data["fs_values"])
        sig_rows.append({
            "signal": sig,
            "n_segments_present": data["count"],
            "fs_median": np.median(fs_arr),
            "fs_min": fs_arr.min(),
            "fs_max": fs_arr.max(),
            "units": ", ".join(data["units"]),
        })
    df_signals = pd.DataFrame(sig_rows)
    if len(df_signals) > 0:
        df_signals = df_signals.sort_values("n_segments_present", ascending=False)

    return df_records, df_signals


# ─── 2. PARSE MIMIC-III NUMERIC SUBSET HEADERS ──────────────────────────────




# ─── 3. LOAD CLINICAL DATA SUMMARIES ────────────────────────────────────────

def load_clinical_summaries():
    """Load pre-computed clinical dataset statistics."""
    meta = pd.read_csv(RESULTS / "clinical_metadata_audit.csv")
    quality = pd.read_csv(RESULTS / "quality_sample_summary.csv")
    sampling = pd.read_csv(RESULTS / "sampling_rates_summary.csv")
    duration = pd.read_csv(RESULTS / "session_duration_stats.csv")
    return meta, quality, sampling, duration


# ─── 4. BUILD FORMAL COMPARISON TABLE ───────────────────────────────────────

def build_comparison_table(clinical_meta, clinical_sampling, clinical_duration,
                           mimic_wave_records, mimic_wave_signals,
                           mimic4_props=None):
    """Build a structured side-by-side comparison table.
    
    mimic4_props: dict signal_name → proportion (0-1) from the full MIMIC-IV WDB.
    """
    if mimic4_props is None:
        mimic4_props = {}

    # Clinical stats
    n_clinical = len(clinical_meta)
    total_hours_clinical = clinical_meta["duration_min"].sum() / 60
    median_dur_clinical = clinical_meta["duration_min"].median()
    n_boxes = clinical_meta["box"].nunique() if "box" in clinical_meta.columns else "N/A"

    # Signal availability from clinical metadata
    sig_cols = [c for c in clinical_meta.columns if c.startswith("has_")]
    clinical_avail = {}
    for c in sig_cols:
        sig_name = c.replace("has_", "").upper()
        pct = clinical_meta[c].mean() * 100 if clinical_meta[c].dtype == bool else clinical_meta[c].astype(float).mean() * 100
        clinical_avail[sig_name] = round(pct, 1)

    # Clinical sampling rates (top waveforms)
    clinical_fs = {}
    if "track_name" in clinical_sampling.columns and "median_srate" in clinical_sampling.columns:
        for _, row in clinical_sampling.iterrows():
            clinical_fs[row["track_name"]] = row["median_srate"]

    # MIMIC-IV waveform stats (use full DB counts, not local subset)
    n_mimic_wave = MIMIC4_TOTAL_RECORDS
    # Duration stats from local subset only (3 downloaded records)
    n_local = len(mimic_wave_records)
    total_hours_mimic = mimic_wave_records["duration_hours"].sum() if n_local > 0 else 0
    median_dur_mimic = (mimic_wave_records["duration_hours"].median() * 60) if n_local > 0 else 0

    # We no longer include MIMIC-III numeric subset in the comparison

    # Signal mapping: clinical signal name → MIMIC equivalent
    # Add ICP explicitly and map to MIMIC waveform where present
    signal_map = {
        "ECG":  {"mimic_wave": "II"},
        "SPO2": {"mimic_wave": "Pleth"},
        "ART":  {"mimic_wave": "ABP"},
        "RESP": {"mimic_wave": "Resp"},
        "CO2":  {"mimic_wave": "CO2"},
        "ICP":  {"mimic_wave": "ICP"},
        "NIBP": {"mimic_wave": None},
        "TEMP": {"mimic_wave": None},
        "BIS":  {"mimic_wave": None},
    }

    COL_WAVE = "MIMIC-IV Waveform (N=200)"

    rows = []
    # General characteristics row
    rows.append({
        "Characteristic": "Total records/files",
        "Clinical (This Study)": f"{n_clinical:,}",
        COL_WAVE: f"{n_mimic_wave} ({MIMIC4_TOTAL_PATIENTS} patients)",
    })
    rows.append({
        "Characteristic": "Total monitoring hours",
        "Clinical (This Study)": f"{total_hours_clinical:,.1f}",
        COL_WAVE: f"{total_hours_mimic:,.1f}*",
    })
    rows.append({
        "Characteristic": "Median session (min)",
        "Clinical (This Study)": f"{median_dur_clinical:.1f}",
        COL_WAVE: f"{median_dur_mimic:.1f}*",
    })
    rows.append({
        "Characteristic": "Monitoring sources",
        "Clinical (This Study)": f"{n_boxes} bedside monitors (GE CARESCAPE)",
        COL_WAVE: "BIDMC ICU bedside monitors",
    })
    rows.append({
        "Characteristic": "Data format",
        "Clinical (This Study)": ".vital (VitalDB binary)",
        COL_WAVE: "WFDB multi-segment (.hea/.dat)",
    })

    # ECG sampling rate
    ecg_fs_clinical = clinical_fs.get("ECG_WAV", clinical_fs.get("ECG_II", ""))
    ecg_fs_mimic = mimic_wave_signals[mimic_wave_signals["signal"] == "II"]["fs_median"].values
    ecg_fs_str = f"{ecg_fs_mimic[0]:.0f}" if len(ecg_fs_mimic) > 0 else "~250"
    rows.append({
        "Characteristic": "ECG sampling rate (Hz)",
        "Clinical (This Study)": f"{ecg_fs_clinical}" if ecg_fs_clinical else "500",
        COL_WAVE: ecg_fs_str,
    })

    # Pleth/SpO2 sampling rate
    pleth_fs_mimic = mimic_wave_signals[mimic_wave_signals["signal"] == "Pleth"]["fs_median"].values
    pleth_fs_str = f"{pleth_fs_mimic[0]:.0f}" if len(pleth_fs_mimic) > 0 else "~125"
    rows.append({
        "Characteristic": "SpO2/Pleth sampling rate (Hz)",
        "Clinical (This Study)": "125",
        COL_WAVE: pleth_fs_str,
    })

    # Signal availability comparison — use real proportions from full MIMIC-IV WDB
    for sig, mapping in signal_map.items():
        clin_pct = clinical_avail.get(sig, "N/A")
        clin_str = f"{clin_pct}%" if isinstance(clin_pct, (int, float)) else clin_pct

        # MIMIC wave availability from real proportions
        mimic_sig = mapping["mimic_wave"]
        if mimic_sig and mimic4_props:
            prop = mimic4_props.get(mimic_sig, 0)
            if prop > 0:
                n_present = int(round(prop * MIMIC4_TOTAL_RECORDS))
                pct_wave = f"{prop*100:.1f}% ({n_present}/{MIMIC4_TOTAL_RECORDS})"
            else:
                pct_wave = "Not available"
        elif mimic_sig:
            pct_wave = "Not available"
        else:
            pct_wave = "Not available"

        # MIMIC numeric availability
        rows.append({
            "Characteristic": f"Signal: {sig} availability",
            "Clinical (This Study)": clin_str,
            COL_WAVE: pct_wave,
        })

    # Unique signals — use count from full proportions CSV if available
    n_tracks_clinical = len(clinical_sampling) if len(clinical_sampling) > 0 else "N/A"
    n_wave_sigs = len(mimic4_props) if mimic4_props else (len(mimic_wave_signals) if len(mimic_wave_signals) > 0 else 0)
    rows.append({
        "Characteristic": "Unique signal types",
        "Clinical (This Study)": str(n_tracks_clinical),
        COL_WAVE: str(n_wave_sigs),
    })

    return pd.DataFrame(rows)


# ─── 5. VISUALIZATION ───────────────────────────────────────────────────────

def plot_signal_comparison(clinical_meta, mimic_wave_signals,
                           mimic4_props=None):
    """Bar chart comparing signal availability between datasets."""
    if mimic4_props is None:
        mimic4_props = {}

    # Common signal categories
    categories = ["ECG", "SpO2/Pleth", "ABP/ART", "Resp", "CO2", "ICP", "Temperature", "BIS/Neuro"]
    
    # Clinical availability (%)
    sig_map_clinical = {
        "ECG": "has_ecg", "SpO2/Pleth": "has_spo2", "ABP/ART": "has_art",
        "Resp": "has_resp", "CO2": "has_co2", "ICP": "has_icp",
        "Temperature": "has_temp", "BIS/Neuro": "has_bis",
    }
    clinical_pcts = []
    for cat in categories:
        col = sig_map_clinical.get(cat)
        if col and col in clinical_meta.columns:
            clinical_pcts.append(clinical_meta[col].mean() * 100)
        else:
            clinical_pcts.append(0)

    # MIMIC waveform: real proportions from full MIMIC-IV WDB
    # For each category, use the max proportion among representative signals
    wave_map = {"ECG": ["II", "V", "aVR", "III", "I"],
                "SpO2/Pleth": ["Pleth"],
                "ABP/ART": ["ABP", "ART"],
                "Resp": ["Resp"],
                "CO2": ["CO2"], "ICP": ["ICP"], "Temperature": [], "BIS/Neuro": []}
    wave_pcts = []
    for cat in categories:
        sigs = wave_map.get(cat, [])
        if sigs and mimic4_props:
            best_prop = max(mimic4_props.get(s, 0) for s in sigs)
            wave_pcts.append(best_prop * 100)
        else:
            wave_pcts.append(0)

    # No MIMIC-III numeric subset in this comparison

    # Plot
    x = np.arange(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, clinical_pcts, width, label="Clinical (This Study)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, wave_pcts, width, label="MIMIC-IV Waveform", color="#FF9800", alpha=0.85)

    ax.set_ylabel("Signal Availability (%)")
    ax.set_title("Signal Availability Comparison: Clinical Dataset vs MIMIC")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 115)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "comparison_bar_signals.png", dpi=150)
    plt.close()
    print(f"  -> Saved comparison_bar_signals.png")


def plot_radar_comparison(clinical_meta, mimic_wave_signals,
                          mimic4_props=None):
    """Radar chart comparing key dimensions between datasets."""
    if mimic4_props is None:
        mimic4_props = {}

    dims = ["ECG", "SpO2", "ABP", "Resp", "CO2", "ICP", "Multi-lead\nECG", "Waveform\nfidelity"]
    
    # Clinical scores (0-1)
    ecg_c = clinical_meta["has_ecg"].mean() if "has_ecg" in clinical_meta.columns else 0
    spo2_c = clinical_meta["has_spo2"].mean() if "has_spo2" in clinical_meta.columns else 0
    art_c = clinical_meta["has_art"].mean() if "has_art" in clinical_meta.columns else 0
    resp_c = clinical_meta["has_resp"].mean() if "has_resp" in clinical_meta.columns else 0
    co2_c = clinical_meta["has_co2"].mean() if "has_co2" in clinical_meta.columns else 0
    icp_c = clinical_meta["has_icp"].mean() if "has_icp" in clinical_meta.columns else 0
    multi_ecg_c = 1.0  # Has multiple ECG leads (ECG_WAV, ECG_II, etc.)
    wave_fidelity_c = 1.0  # 500 Hz ECG
    clinical_scores = [ecg_c, spo2_c, art_c, resp_c, co2_c, icp_c, multi_ecg_c, wave_fidelity_c]

    # MIMIC wave scores — use real proportions from full MIMIC-IV WDB
    ecg_w = mimic4_props.get("II", 0)
    spo2_w = mimic4_props.get("Pleth", 0)
    art_w = max(mimic4_props.get("ABP", 0), mimic4_props.get("ART", 0))
    resp_w = mimic4_props.get("Resp", 0)
    co2_w = mimic4_props.get("CO2", 0)
    icp_w = mimic4_props.get("ICP", 0)
    # Multi-lead ECG: average of I, III, V, aVR presence
    multi_leads = [mimic4_props.get(s, 0) for s in ["I", "III", "V", "aVR"]]
    multi_ecg_w = np.mean(multi_leads) if multi_leads else 0
    wave_fidelity_w = 0.5  # ~250 Hz ECG (lower than clinical 500 Hz)
    mimic_scores = [ecg_w, spo2_w, art_w, resp_w, co2_w, icp_w, multi_ecg_w, wave_fidelity_w]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    clinical_scores += clinical_scores[:1]
    mimic_scores += mimic_scores[:1]
    numeric_scores = None

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, clinical_scores, "o-", linewidth=2, label="Clinical (This Study)", color="#2196F3")
    ax.fill(angles, clinical_scores, alpha=0.15, color="#2196F3")
    ax.plot(angles, mimic_scores, "s-", linewidth=2, label="MIMIC-IV Waveform", color="#FF9800")
    ax.fill(angles, mimic_scores, alpha=0.15, color="#FF9800")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Dataset Capability Comparison", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    fig.savefig(OUT / "comparison_radar.png", dpi=150)
    plt.close()
    print(f"  -> Saved comparison_radar.png")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("COMPARISON: Clinical Dataset vs MIMIC-IV/III")
    print("=" * 60)

    # 0. Load real MIMIC-IV proportions (full MIMIC-IV Waveform DB v0.1.0)
    print("\n[0/5] Loading MIMIC-IV Waveform proportions (full DB, N=200)...")
    mimic4_prop_df, mimic4_props = load_mimic4_proportions()
    print(f"  Loaded proportions for {len(mimic4_props)} signal types")
    for _, row in mimic4_prop_df.head(10).iterrows():
        print(f"    {row['signal']}: {float(row['proportion'])*100:.1f}% ({row['n_records_present']}/{MIMIC4_TOTAL_RECORDS})")

    # 1. Analyze MIMIC-IV waveforms (local subset — for fs/duration metadata)
    print("\n[1/5] Parsing local MIMIC-IV waveform headers (local subset)...")
    mimic_wave_records, mimic_wave_signals = analyze_mimic_waveforms()
    print(f"  Found {len(mimic_wave_records)} local waveform records, "
          f"{len(mimic_wave_signals)} unique signals in local subset")
    if len(mimic_wave_records) > 0:
        print(f"  Total waveform hours (local): {mimic_wave_records['duration_hours'].sum():.1f}")

    mimic_wave_records.to_csv(OUT / "mimic_waveform_summary.csv", index=False)
    mimic_wave_signals.to_csv(OUT / "mimic_waveform_signals.csv", index=False)
    print(f"  -> Saved mimic_waveform_summary.csv, mimic_waveform_signals.csv")

    # 2. (skipped) MIMIC-III numeric subset removed from comparison
    print("\n[2/5] MIMIC-III numeric subset skipped (not included in this comparison)")

    # 3. Load clinical summaries
    print("\n[3/5] Loading clinical dataset summaries...")
    clinical_meta, clinical_quality, clinical_sampling, clinical_duration = load_clinical_summaries()
    # Ensure respiratory and ICP presence flags exist in clinical metadata
    if "has_resp" not in clinical_meta.columns:
        clinical_meta["has_resp"] = clinical_meta["tracks"].str.contains(r"\bRR\b|RESP|VENT_RR", case=False, regex=True).astype(int)
    if "has_icp" not in clinical_meta.columns:
        clinical_meta["has_icp"] = clinical_meta["tracks"].str.contains(r"\bICP\b", case=False, regex=True).astype(int)
    print(f"  Clinical: {len(clinical_meta)} files, "
          f"{clinical_meta['duration_min'].sum() / 60:.1f} hours, "
          f"{len(clinical_sampling)} track types")

    # 4. Build comparison table
    print("\n[4/5] Building formal comparison table...")
    comp_table = build_comparison_table(
        clinical_meta, clinical_sampling, clinical_duration,
        mimic_wave_records, mimic_wave_signals,
        mimic4_props=mimic4_props)
    comp_table.to_csv(OUT / "comparison_table.csv", index=False)
    print(f"  -> Saved comparison_table.csv ({len(comp_table)} rows)")
    print("\n  Comparison Table:")
    col_wave = "MIMIC-IV Waveform (N=200)"
    for _, row in comp_table.iterrows():
        print(f"    {row['Characteristic']}: Clinical={row['Clinical (This Study)']} | "
              f"MIMIC-Wave={row[col_wave]}")

    # 5. Visualizations
    print("\n[5/5] Generating comparison visualizations...")
    plot_signal_comparison(clinical_meta, mimic_wave_signals, mimic4_props)
    plot_radar_comparison(clinical_meta, mimic_wave_signals, mimic4_props)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print(f"Outputs saved to: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
