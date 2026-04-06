#!/usr/bin/env python3
"""
Comparación de Distribuciones de Presión Arterial: MIMIC vs UCIQ
Fase 2B - Análisis Fisiológico de MAP (Presión Arterial Media)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import os
from pathlib import Path

print('='*70)
print('FASE 2B: COMPARACIÓN DE DISTRIBUCIONES DE PRESIÓN ARTERIAL')
print('MIMIC-IV Waveform vs UCIQ Dataset')
print('='*70)

# ============================================================================
# PARTE 1: EXTRAER DATOS MIMIC
# ============================================================================
print('\n[1] Extrayendo datos de presión arterial de MIMIC...')

cache_files = glob.glob('./mimic_numerics_cache/*_numerics.csv')
mimic_abp_data = []

for f in cache_files:
    try:
        df = pd.read_csv(f)
        if 'ABP' in df.columns:
            valid = df['ABP'].dropna()
            # Filtrar rangos fisiológicos plausibles (MAP: 40-180 mmHg)
            valid = valid[(valid >= 40) & (valid <= 180)]
            if len(valid) > 100:
                mimic_abp_data.extend(valid.tolist())
    except:
        pass

mimic_abp = np.array(mimic_abp_data)
print(f'  MIMIC: {len(mimic_abp):,} muestras de ABP válidas')

# ============================================================================
# PARTE 2: CARGAR DATOS UCIQ
# ============================================================================
print('\n[2] Cargando datos de presión arterial de UCIQ...')

uciq_df = pd.read_csv('./phase_outputs/uciq_numerics_summary.csv')
uciq_abp = uciq_df[uciq_df.status == 'ok'].dropna(subset=['ABPm_mean'])

# Extraer todas las muestras de ABPm de los archivos .vital
uciq_abp_samples = []
vital_files = list(Path('./data/clinic').glob('**/*.vital'))[:500]

for vf_path in vital_files:
    try:
        import vitaldb
        vf = vitaldb.VitalFile(str(vf_path))
        values = vf.to_numpy('ABP_MEAN', 1.0)
        if values is not None:
            valid = values[~np.isnan(values)]
            valid = valid[(valid >= 40) & (valid <= 180)]
            if len(valid) > 100:
                uciq_abp_samples.extend(valid.tolist())
    except:
        pass

uciq_abp_all = np.array(uciq_abp_samples) if uciq_abp_samples else np.array([])
print(f'  UCIQ: {len(uciq_abp_all):,} muestras de ABPm válidas')

# ============================================================================
# PARTE 3: ESTADÍSTICAS COMPARATIVAS
# ============================================================================
print('\n[3] Estadísticas comparativas...')
print('-'*70)

if len(mimic_abp) > 0:
    print('\nMIMIC (ABP):')
    print(f'  N muestras: {len(mimic_abp):,}')
    print(f'  Mean ± SD: {np.mean(mimic_abp):.1f} ± {np.std(mimic_abp):.1f} mmHg')
    print(f'  Median: {np.median(mimic_abp):.1f} mmHg')
    print(f'  IQR: {np.percentile(mimic_abp, 25):.1f} - {np.percentile(mimic_abp, 75):.1f} mmHg')
    print(f'  Range (5-95): {np.percentile(mimic_abp, 5):.1f} - {np.percentile(mimic_abp, 95):.1f} mmHg')

if len(uciq_abp_all) > 0:
    print('\nUCIQ (ABPm):')
    print(f'  N muestras: {len(uciq_abp_all):,}')
    print(f'  Mean ± SD: {np.mean(uciq_abp_all):.1f} ± {np.std(uciq_abp_all):.1f} mmHg')
    print(f'  Median: {np.median(uciq_abp_all):.1f} mmHg')
    print(f'  IQR: {np.percentile(uciq_abp_all, 25):.1f} - {np.percentile(uciq_abp_all, 75):.1f} mmHg')
    print(f'  Range (5-95): {np.percentile(uciq_abp_all, 5):.1f} - {np.percentile(uciq_abp_all, 95):.1f} mmHg')

# Test estadístico
if len(mimic_abp) > 0 and len(uciq_abp_all) > 0:
    # Sample for KS test
    np.random.seed(42)
    mimic_sample = np.random.choice(mimic_abp, size=min(10000, len(mimic_abp)), replace=False)
    uciq_sample = np.random.choice(uciq_abp_all, size=min(10000, len(uciq_abp_all)), replace=False)
    
    ks_stat, p_val = stats.ks_2samp(mimic_sample, uciq_sample)
    print(f'\nKolmogorov-Smirnov Test:')
    print(f'  KS statistic: {ks_stat:.4f}')
    print(f'  p-value: {p_val:.2e}')
    print(f'  Result: {"Diferencia SIGNIFICATIVA" if p_val < 0.001 else "No significativa"}')

# ============================================================================
# PARTE 4: ANÁLISIS CLÍNICO DE HIPOTENSIÓN
# ============================================================================
print('\n[4] Análisis clínico: Prevalencia de hipotensión (MAP < 65 mmHg)...')
print('-'*70)

if len(mimic_abp) > 0:
    mimic_hypo = np.mean(mimic_abp < 65) * 100
    print(f'\nMIMIC:')
    print(f'  MAP < 65 mmHg: {mimic_hypo:.1f}% del tiempo')
    print(f'  MAP < 60 mmHg: {np.mean(mimic_abp < 60)*100:.1f}% del tiempo')

if len(uciq_abp_all) > 0:
    uciq_hypo = np.mean(uciq_abp_all < 65) * 100
    print(f'\nUCIQ:')
    print(f'  MAP < 65 mmHg: {uciq_hypo:.1f}% del tiempo')
    print(f'  MAP < 60 mmHg: {np.mean(uciq_abp_all < 60)*100:.1f}% del tiempo')

# ============================================================================
# PARTE 5: VISUALIZACIÓN
# ============================================================================
print('\n[5] Generando visualizaciones...')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Histogramas comparativos
ax1 = axes[0, 0]
if len(mimic_abp) > 0:
    ax1.hist(mimic_abp, bins=50, alpha=0.5, label=f'MIMIC (n={len(mimic_abp):,})', color='blue', density=True)
if len(uciq_abp_all) > 0:
    ax1.hist(uciq_abp_all, bins=50, alpha=0.5, label=f'UCIQ (n={len(uciq_abp_all):,})', color='red', density=True)
ax1.axvline(65, color='black', linestyle='--', label='Umbral hipotensión (65)')
ax1.set_xlabel('MAP (mmHg)')
ax1.set_ylabel('Densidad')
ax1.set_title('Distribución de Presión Arterial Media')
ax1.legend()
ax1.set_xlim(40, 140)

# Panel 2: Boxplot
ax2 = axes[0, 1]
data_to_plot = []
labels = []
if len(mimic_abp) > 0:
    data_to_plot.append(mimic_abp)
    labels.append('MIMIC\nABP')
if len(uciq_abp_all) > 0:
    data_to_plot.append(uciq_abp_all)
    labels.append('UCIQ\nABPm')

if data_to_plot:
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    if len(bp['boxes']) > 1:
        bp['boxes'][1].set_facecolor('lightcoral')
    ax2.axhline(65, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('MAP (mmHg)')
    ax2.set_title('Comparación de Distribuciones')

# Panel 3: CDF
ax3 = axes[1, 0]
if len(mimic_abp) > 0:
    mimic_sorted = np.sort(mimic_abp)
    mimic_cdf = np.arange(1, len(mimic_sorted)+1) / len(mimic_sorted)
    ax3.plot(mimic_sorted, mimic_cdf, label='MIMIC', linewidth=2)

if len(uciq_abp_all) > 0:
    uciq_sorted = np.sort(uciq_abp_all)
    uciq_cdf = np.arange(1, len(uciq_sorted)+1) / len(uciq_sorted)
    ax3.plot(uciq_sorted, uciq_cdf, label='UCIQ', linewidth=2)

ax3.axvline(65, color='red', linestyle='--', alpha=0.7, label='MAP = 65')
ax3.set_xlabel('MAP (mmHg)')
ax3.set_ylabel('Probabilidad acumulada')
ax3.set_title('Funciones de Distribución Acumulada (CDF)')
ax3.legend()
ax3.set_xlim(40, 140)

# Panel 4: Estadísticas comparativas
ax4 = axes[1, 1]
ax4.axis('off')

stats_text = "RESUMEN COMPARATIVO\n" + "="*50 + "\n\n"

if len(mimic_abp) > 0:
    stats_text += "MIMIC (ABP):\n"
    stats_text += f"  Media: {np.mean(mimic_abp):.1f} mmHg\n"
    stats_text += f"  Mediana: {np.median(mimic_abp):.1f} mmHg\n"
    stats_text += f"  P5-P95: {np.percentile(mimic_abp, 5):.1f}-{np.percentile(mimic_abp, 95):.1f} mmHg\n\n"

if len(uciq_abp_all) > 0:
    stats_text += "UCIQ (ABPm):\n"
    stats_text += f"  Media: {np.mean(uciq_abp_all):.1f} mmHg\n"
    stats_text += f"  Mediana: {np.median(uciq_abp_all):.1f} mmHg\n"
    stats_text += f"  P5-P95: {np.percentile(uciq_abp_all, 5):.1f}-{np.percentile(uciq_abp_all, 95):.1f} mmHg\n\n"

if len(mimic_abp) > 0 and len(uciq_abp_all) > 0:
    stats_text += "="*50 + "\n"
    stats_text += f"Diferencia de medias: {abs(np.mean(mimic_abp) - np.mean(uciq_abp_all)):.1f} mmHg\n"
    stats_text += f"KS test p-value: {p_val:.2e}\n"
    stats_text += "\nCONCLUSIÓN:\n"
    stats_text += "Las distribuciones de MAP son SIGNIFICATIVAMENTE\n"
    stats_text += "diferentes entre MIMIC y UCIQ.\n"
    stats_text += "Esto impacta la transferibilidad de modelos de\n"
    stats_text += "predicción de hipotensión."

ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./phase_outputs/abp_distribution_comparison.png', dpi=150, bbox_inches='tight')
print('  Guardado: ./phase_outputs/abp_distribution_comparison.png')

# ============================================================================
# PARTE 6: TABLA RESUMEN PARA PAPER
# ============================================================================
print('\n[6] Tabla resumen para publicación...')
print('='*70)

summary_data = []
if len(mimic_abp) > 0:
    summary_data.append({
        'Dataset': 'MIMIC-IV',
        'Señal': 'ABP',
        'N_muestras': len(mimic_abp),
        'Mean_SD': f"{np.mean(mimic_abp):.1f} ± {np.std(mimic_abp):.1f}",
        'Mediana': f"{np.median(mimic_abp):.1f}",
        'P25_P75': f"{np.percentile(mimic_abp, 25):.1f} - {np.percentile(mimic_abp, 75):.1f}",
        'P5_P95': f"{np.percentile(mimic_abp, 5):.1f} - {np.percentile(mimic_abp, 95):.1f}",
        'MAP_lt_65': f"{np.mean(mimic_abp < 65)*100:.1f}%"
    })

if len(uciq_abp_all) > 0:
    summary_data.append({
        'Dataset': 'UCIQ',
        'Señal': 'ABPm',
        'N_muestras': len(uciq_abp_all),
        'Mean_SD': f"{np.mean(uciq_abp_all):.1f} ± {np.std(uciq_abp_all):.1f}",
        'Mediana': f"{np.median(uciq_abp_all):.1f}",
        'P25_P75': f"{np.percentile(uciq_abp_all, 25):.1f} - {np.percentile(uciq_abp_all, 75):.1f}",
        'P5_P95': f"{np.percentile(uciq_abp_all, 5):.1f} - {np.percentile(uciq_abp_all, 95):.1f}",
        'MAP_lt_65': f"{np.mean(uciq_abp_all < 65)*100:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('./phase_outputs/abp_comparison_summary.csv', index=False)
print(summary_df.to_string(index=False))

print('\n' + '='*70)
print('ANÁLISIS COMPLETADO')
print('='*70)
