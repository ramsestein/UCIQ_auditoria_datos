"""
Análisis D: Catálogo de problemas técnicos y lecciones aprendidas.
Sintetiza todos los hallazgos de los análisis previos en un catálogo estructurado
de problemas técnicos encontrados durante la recolección de datos de monitorización UCI,
con impacto cuantificado, señales afectadas, y recomendaciones.
Diseñado para que otros investigadores puedan anticipar y resolver problemas similares.
"""
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "results_auditory"
FINAL_DIR = os.path.join(OUTPUT_DIR, "results_final")


def load_data():
    """Carga todos los CSVs necesarios."""
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "clinical_metadata_audit.csv"))
    quality = pd.read_csv(os.path.join(OUTPUT_DIR, "quality_sample_summary.csv"))
    artifacts = pd.read_csv(os.path.join(FINAL_DIR, "artifact_detection_summary.csv"))
    artifact_box = pd.read_csv(os.path.join(FINAL_DIR, "artifact_by_box.csv"))
    gaps = pd.read_csv(os.path.join(FINAL_DIR, "box_gap_summary.csv"))
    completeness = pd.read_csv(os.path.join(FINAL_DIR, "completeness_by_box.csv"))
    duration = pd.read_csv(os.path.join(OUTPUT_DIR, "session_duration_stats.csv"))
    sampling = pd.read_csv(os.path.join(OUTPUT_DIR, "sampling_rates_summary.csv"))
    return meta, quality, artifacts, artifact_box, gaps, completeness, duration, sampling


def analyze_issue_1_session_fragmentation(meta, gaps):
    """P1: Fragmentación excesiva de sesiones."""
    total_files = len(meta)
    micro = meta[meta['duration_min'] < 5] if 'duration_min' in meta.columns else pd.DataFrame()
    short = meta[meta['duration_min'] < 30] if 'duration_min' in meta.columns else pd.DataFrame()

    # Gaps entre sesiones
    total_gaps = gaps['n_sessions'].sum() - len(gaps)  # gaps = sessions - 1 per box
    long_gaps = gaps['long'].sum() + gaps['very_long'].sum()

    worst_gap_box = gaps.loc[gaps['mean_gap_hours'].idxmax()]

    return {
        'id': 'P1',
        'category': 'Data Continuity',
        'title': 'Session Fragmentation',
        'description': 'Monitoring sessions are split into many short files instead of continuous recordings. '
                       'Each patient reconnection or monitor restart creates a new .vital file.',
        'impact': f'{len(micro)}/{total_files} files ({100*len(micro)/total_files:.1f}%) are <5 min; '
                  f'{len(short)}/{total_files} ({100*len(short)/total_files:.1f}%) are <30 min',
        'affected_signals': 'All signals',
        'worst_box': f'{worst_gap_box["box"]} (mean gap: {worst_gap_box["mean_gap_hours"]:.1f}h)',
        'total_long_gaps': int(long_gaps),
        'recommendation': 'Implement session merging algorithms based on temporal proximity. '
                          'Consider grouping files within 30-minute gaps as a single clinical session.',
        'lesson': 'Monitor auto-save intervals and nurse workflow patterns cause natural fragmentation. '
                  'Plan for post-hoc session reconstruction in the data pipeline.'
    }


def analyze_issue_2_signal_availability(completeness):
    """P2: Disponibilidad variable de señales avanzadas."""
    # ECG y SpO2 están ~99%, pero otras señales son mucho más bajas
    signals = ['has_art', 'has_co2', 'has_bis', 'has_neuro', 'has_hemo', 'has_vent', 'has_temp']
    means = {col: completeness[col].mean() for col in signals}
    worst = min(means, key=means.get)
    best_advanced = max((k for k in means if means[k] < 90), key=means.get, default=worst)

    return {
        'id': 'P2',
        'category': 'Signal Availability',
        'title': 'Low Availability of Advanced Monitoring Signals',
        'description': 'While basic signals (ECG, SpO2) are near-universal (>99%), '
                       'advanced monitoring signals have much lower availability.',
        'impact': '; '.join(f'{s.replace("has_","")}={means[s]:.1f}%' for s in signals),
        'affected_signals': ', '.join(s.replace('has_', '').upper() for s in signals if means[s] < 50),
        'worst_signal': f'{worst.replace("has_","")} ({means[worst]:.1f}%)',
        'best_advanced': f'{best_advanced.replace("has_","")} ({means[best_advanced]:.1f}%)',
        'recommendation': 'Accept that advanced signals will only be available in subsets of patients. '
                          'Design analyses to handle variable signal availability gracefully.',
        'lesson': 'Signal availability depends on clinical indication (e.g., ICP only with neurosurgery, '
                  'ventilator only with mechanical ventilation). This is not a technical failure but '
                  'reflects real clinical practice.'
    }


def analyze_issue_3_artifacts(artifacts):
    """P3: Artefactos y valores fisiológicamente imposibles."""
    high_artifact = artifacts[artifacts['overall_pct_artifact'] > 0.5].sort_values(
        'overall_pct_artifact', ascending=False)
    low_artifact = artifacts[artifacts['overall_pct_artifact'] <= 0.01]

    return {
        'id': 'P3',
        'category': 'Data Quality',
        'title': 'Physiologically Impossible Values (Artifacts)',
        'description': 'Certain signals contain values outside physiological limits, '
                       'requiring artifact detection before analysis.',
        'impact': f'{len(high_artifact)} signals with >0.5% artifact rate; '
                  f'{len(low_artifact)} signals with <0.01% artifact rate',
        'affected_signals': ', '.join(high_artifact['signal_desc'].tolist()) if len(high_artifact) > 0 else 'None',
        'worst_signals': '; '.join(
            f'{r["signal_desc"]}: {r["overall_pct_artifact"]:.2f}%'
            for _, r in high_artifact.head(5).iterrows()
        ),
        'cleanest_signals': '; '.join(
            f'{r["signal_desc"]}: {r["overall_pct_artifact"]:.3f}%'
            for _, r in low_artifact.head(5).iterrows()
        ),
        'recommendation': 'Apply physiological range filters before any statistical analysis. '
                          'Temperature and ICP require special attention (>2% artifact rate). '
                          'Spike detection is critical for arterial pressure signals.',
        'lesson': 'Artifact rates vary dramatically by signal type. Temperature sensors are '
                  'particularly unreliable when disconnected. Arterial pressure shows spike artifacts '
                  'from catheter flushes and zeroing.'
    }


def analyze_issue_4_temperature(artifacts):
    """P4: Problema específico de temperatura."""
    temp_rows = artifacts[artifacts['signal'].isin(['TEMP', 'BT_SKIN'])]
    if len(temp_rows) == 0:
        return None

    return {
        'id': 'P4',
        'category': 'Sensor Specific',
        'title': 'Temperature Sensor Unreliability',
        'description': 'Temperature signals (core and skin) show the highest artifact rates '
                       'of all monitored signals, primarily from sensor disconnection.',
        'impact': '; '.join(
            f'{r["signal_desc"]}: {r["overall_pct_artifact"]:.1f}% artifacts, '
            f'only {int(r["n_files"])} files'
            for _, r in temp_rows.iterrows()
        ),
        'affected_signals': 'TEMP, BT_SKIN',
        'recommendation': 'Temperature data requires aggressive filtering. Values <30°C or >42°C '
                          'are almost certainly sensor errors. Consider excluding files with >20% '
                          'temperature artifacts.',
        'lesson': 'Temperature probes are frequently disconnected during patient care activities '
                  '(bathing, repositioning). The low availability (1.7%) combined with high artifact '
                  'rate makes temperature one of the least reliable continuous signals.'
    }


def analyze_issue_5_spo2_detection(meta):
    """P5: Problema de detección de SpO2 en metadatos."""
    # SpO2 tracks have multiple naming conventions
    return {
        'id': 'P5',
        'category': 'Metadata',
        'title': 'SpO2 Track Name Variability',
        'description': 'SpO2 signals were initially under-detected in metadata extraction because '
                       'the track names varied across devices. Names include PLETH_SAT_O2, '
                       'SpO2, SAT_O2, PLETH_S, etc.',
        'impact': 'Initial extraction missed ~6% of SpO2 signals until pattern matching was expanded',
        'affected_signals': 'SpO2 / PLETH_SAT_O2',
        'recommendation': 'Use broad regex pattern matching for signal detection: '
                          'r"spo2|pleth|sat|sat_o2|pleth_sat|pleth_s". '
                          'Validate signal detection against manual review of a sample.',
        'lesson': 'Different monitor firmware versions and configurations may use different '
                  'internal track names for the same clinical signal. Always validate metadata '
                  'extraction against the raw files.'
    }


def analyze_issue_6_data_quality_variance(quality):
    """P6: Varianza de calidad entre archivos."""
    low_quality = quality[quality['pct_non_nan'] < 50]
    flatline = quality[quality['flatline_pct'] > 10]

    return {
        'id': 'P6',
        'category': 'Data Quality',
        'title': 'High Variance in Data Quality Across Files',
        'description': 'Signal quality varies dramatically between files. Some files have '
                       'excellent data (>95% non-NaN), while others are mostly empty.',
        'impact': f'{len(low_quality)}/{len(quality)} files ({100*len(low_quality)/len(quality):.1f}%) '
                  f'have <50% valid data; {len(flatline)} files ({100*len(flatline)/len(quality):.1f}%) '
                  f'have >10% flatline',
        'affected_signals': 'All signals',
        'mean_quality': f'{quality["pct_non_nan"].mean():.1f}%',
        'median_quality': f'{quality["pct_non_nan"].median():.1f}%',
        'recommendation': 'Implement quality-based file filtering before analysis. '
                          'Suggested minimum: >50% non-NaN data AND <10% flatline. '
                          'This excludes monitor-on-but-disconnected periods.',
        'lesson': 'A significant fraction of .vital files capture periods where the monitor '
                  'is powered on but not connected to a patient. Quality filtering is essential '
                  'before any clinical analysis.'
    }


def analyze_issue_7_box_downtime(meta, gaps):
    """P7: Downtime irregular de boxes."""
    box_stats = meta.groupby('box').agg(
        n_files=('filename', 'count'),
        first=('date', 'min'),
        last=('date', 'max')
    ).reset_index()

    # Identify boxes with very low consistency
    gaps_sorted = gaps.sort_values('mean_gap_hours', ascending=False)
    worst = gaps_sorted.head(3)

    return {
        'id': 'P7',
        'category': 'Infrastructure',
        'title': 'Irregular Box Availability and Downtime',
        'description': 'Not all ICU monitoring boxes operate continuously. Some have extended '
                       'periods of inactivity due to hardware failures, maintenance, or '
                       'redeployment.',
        'impact': f'{len(box_stats)} boxes total; mean gap ranges from '
                  f'{gaps["mean_gap_hours"].min():.1f}h to {gaps["mean_gap_hours"].max():.1f}h',
        'affected_signals': 'All signals (infrastructure-level)',
        'worst_boxes': '; '.join(
            f'{r["box"]} (mean gap: {r["mean_gap_hours"]:.1f}h, '
            f'{r["very_long"]} very long gaps)'
            for _, r in worst.iterrows()
        ),
        'recommendation': 'Identify and document continuous-operation boxes (≥50% months active) '
                          'for temporal trend analyses. Use all boxes for cross-sectional analyses '
                          'but only continuous boxes for longitudinal studies.',
        'lesson': 'Real-world ICU deployments have irregular uptime. Hardware failures, network '
                  'issues, and clinical workflow disruptions all cause data gaps that cannot '
                  'be prevented. Design studies to be robust to missing periods.'
    }


def analyze_issue_8_nibp_anomaly(artifacts):
    """P8: Anomalía NIBP_SYS."""
    nibp_row = artifacts[artifacts['signal'] == 'NIBP_SYS']
    if len(nibp_row) == 0:
        return None
    nibp = nibp_row.iloc[0]
    if nibp['mean_signal_value'] > 500:  # Clearly anomalous mean
        return {
            'id': 'P8',
            'category': 'Data Quality',
            'title': 'NIBP Systolic Anomalous Mean Values',
            'description': 'Non-invasive blood pressure systolic readings show an anomalous '
                           f'mean value of {nibp["mean_signal_value"]:.1f} mmHg, far exceeding '
                           'physiological norms (normal ~120 mmHg). This suggests occasional '
                           'extreme outliers or encoding errors in the NIBP channel.',
            'impact': f'{int(nibp["n_files"])} files affected; mean value {nibp["mean_signal_value"]:.1f} mmHg',
            'affected_signals': 'NIBP_SYS',
            'recommendation': 'Apply strict range filtering (40-280 mmHg) to NIBP systolic values. '
                              'Investigate whether extreme values come from measurement errors '
                              'during cuff inflation or firmware encoding issues.',
            'lesson': 'NIBP is intermittent by nature (measured every 5-15 minutes) and its '
                      'numeric channel may contain cuff pressure or error codes alongside '
                      'actual blood pressure readings.'
        }
    return None


def analyze_issue_9_arterial_spikes(artifacts):
    """P9: Spikes en presión arterial."""
    art_signals = artifacts[artifacts['total_spikes'] > 100]
    if len(art_signals) == 0:
        return None

    return {
        'id': 'P9',
        'category': 'Data Quality',
        'title': 'Arterial Pressure Spike Artifacts',
        'description': 'Invasive arterial pressure signals show significant spike artifacts — '
                       'sudden jumps in consecutive samples that exceed physiological possibility.',
        'impact': '; '.join(
            f'{r["signal_desc"]}: {int(r["total_spikes"])} spikes across {int(r["n_files"])} files'
            for _, r in art_signals.iterrows()
        ),
        'affected_signals': ', '.join(art_signals['signal_desc'].tolist()),
        'recommendation': 'Apply median filtering or spike rejection (>50 mmHg/s change) to '
                          'arterial waveform-derived numerics before analysis. Spikes often '
                          'correspond to catheter flushing, blood sampling, or transducer zeroing.',
        'lesson': 'Arterial catheter maintenance procedures create characteristic spike artifacts. '
                  'These are predictable and filterable but must be accounted for in any '
                  'hemodynamic stability analysis.'
    }


def generate_catalog(issues):
    """Genera CSV y visualización del catálogo."""
    rows = []
    for iss in issues:
        if iss is None:
            continue
        rows.append({
            'issue_id': iss['id'],
            'category': iss['category'],
            'title': iss['title'],
            'description': iss['description'],
            'impact': iss['impact'],
            'affected_signals': iss.get('affected_signals', ''),
            'recommendation': iss['recommendation'],
            'lesson_learned': iss['lesson'],
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "technical_issues_catalog.csv")
    df.to_csv(out_path, index=False)
    print(f"Catalog saved: {out_path} ({len(df)} issues)")

    # Visualización: resumen por categoría
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cat_counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(cat_counts.index, cat_counts.values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_xlabel('Number of Issues')
    ax.set_title('Technical Issues by Category')
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'viz_technical_issues.png'), dpi=150)
    plt.close()
    print("Visualization saved: viz_technical_issues.png")

    # Print summary
    print("\n" + "="*70)
    print("TECHNICAL ISSUES CATALOG")
    print("="*70)
    for _, row in df.iterrows():
        print(f"\n[{row['issue_id']}] {row['title']} ({row['category']})")
        print(f"  Impact: {row['impact'][:120]}")
        print(f"  Lesson: {row['lesson_learned'][:120]}")
    print("\n" + "="*70)

    return df


def run():
    print("Loading analysis data...")
    meta, quality, artifacts, artifact_box, gaps, completeness, duration, sampling = load_data()

    print("Analyzing technical issues...")
    issues = [
        analyze_issue_1_session_fragmentation(meta, gaps),
        analyze_issue_2_signal_availability(completeness),
        analyze_issue_3_artifacts(artifacts),
        analyze_issue_4_temperature(artifacts),
        analyze_issue_5_spo2_detection(meta),
        analyze_issue_6_data_quality_variance(quality),
        analyze_issue_7_box_downtime(meta, gaps),
        analyze_issue_8_nibp_anomaly(artifacts),
        analyze_issue_9_arterial_spikes(artifacts),
    ]

    catalog = generate_catalog(issues)
    return catalog


if __name__ == '__main__':
    run()
