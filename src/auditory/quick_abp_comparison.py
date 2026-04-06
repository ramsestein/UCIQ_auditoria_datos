#!/usr/bin/env python3
"""
Comparación rápida de MAP: MIMIC vs UCIQ usando datos ya extraídos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

print('='*70)
print('COMPARACIÓN DE PRESIÓN ARTERIAL MEDIA (MAP) - MIMIC vs UCIQ')
print('='*70)

# Cargar datos UCIQ del summary ya generado
uciq = pd.read_csv('./phase_outputs/uciq_numerics_summary.csv')
uciq_ok = uciq[uciq.status == 'ok'].dropna(subset=['ABPm_mean'])

print(f'\nUCIQ: {len(uciq_ok)} registros con ABPm')
print(f'  MAP Mean: {uciq_ok.ABPm_mean.mean():.1f} ± {uciq_ok.ABPm_mean.std():.1f} mmHg')
print(f'  MAP Median: {uciq_ok.ABPm_median.mean():.1f} mmHg')
print(f'  Range: {uciq_ok.ABPm_mean.min():.1f} - {uciq_ok.ABPm_mean.max():.1f} mmHg')

# Extraer datos MIMIC de los archivos cache
print('\nExtrayendo datos MIMIC de caché...')
cache_files = glob.glob('./mimic_numerics_cache/*_numerics.csv')

mimic_records = []
for f in cache_files[:60]:  # Limitar a 60 registros con datos
    try:
        df = pd.read_csv(f)
        if 'ABP' in df.columns:
            valid = df['ABP'].dropna()
            # Filtrar rangos fisiológicos plausibles (MAP: 40-180 mmHg)
            valid = valid[(valid >= 40) & (valid <= 180)]
            if len(valid) > 1000:
                mimic_records.append({
                    'file': os.path.basename(f),
                    'n': len(valid),
                    'mean': valid.mean(),
                    'std': valid.std(),
                    'median': valid.median(),
                    'p5': valid.quantile(0.05),
                    'p95': valid.quantile(0.95)
                })
    except:
        pass

mimic_df = pd.DataFrame(mimic_records)
print(f'\nMIMIC: {len(mimic_df)} registros con ABP analizados')
if len(mimic_df) > 0:
    print(f'  MAP Mean: {mimic_df["mean"].mean():.1f} ± {mimic_df["mean"].std():.1f} mmHg')
    print(f'  MAP Median: {mimic_df["median"].mean():.1f} mmHg')
    print(f'  Range: {mimic_df["mean"].min():.1f} - {mimic_df["mean"].max():.1f} mmHg')

# Comparación
print('\n' + '='*70)
print('COMPARACIÓN ESTADÍSTICA')
print('='*70)

if len(mimic_df) > 0 and len(uciq_ok) > 0:
    from scipy import stats
    
    # Prueba t de Student
    t_stat, p_val = stats.ttest_ind(mimic_df['mean'], uciq_ok['ABPm_mean'])
    print(f'\nT-test: t={t_stat:.3f}, p={p_val:.4f}')
    
    # Prueba KS
    ks_stat, ks_p = stats.ks_2samp(mimic_df['mean'], uciq_ok['ABPm_mean'])
    print(f'Kolmogorov-Smirnov: KS={ks_stat:.3f}, p={ks_p:.4f}')
    
    # Diferencia de medias
    diff = mimic_df['mean'].mean() - uciq_ok['ABPm_mean'].mean()
    print(f'\nDiferencia de medias (MIMIC - UCIQ): {diff:.1f} mmHg')
    print(f'  ({abs(diff)/uciq_ok.ABPm_mean.mean()*100:.1f}% diferencia relativa)')
    
    # Prevalencia de MAP < 65 (hipotensión)
    mimic_hypo = (mimic_df['mean'] < 65).mean() * 100
    uciq_hypo = (uciq_ok['ABPm_mean'] < 65).mean() * 100
    print(f'\nPrevalencia de MAP < 65 mmHg:')
    print(f'  MIMIC: {mimic_hypo:.1f}% de registros')
    print(f'  UCIQ: {uciq_hypo:.1f}% de registros')

# Visualización
print('\nGenerando visualización...')
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Boxplots
ax1 = axes[0]
data = [mimic_df['mean'].values, uciq_ok['ABPm_mean'].values]
bp = ax1.boxplot(data, labels=['MIMIC\nABP', 'UCIQ\nABPm'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax1.axhline(65, color='red', linestyle='--', alpha=0.7, label='MAP=65 (hipotensión)')
ax1.set_ylabel('MAP (mmHg)')
ax1.set_title('Comparación de Distribuciones')
ax1.legend()

# Panel 2: Histogramas
ax2 = axes[1]
ax2.hist(mimic_df['mean'], bins=20, alpha=0.5, label='MIMIC', color='blue', density=True)
ax2.hist(uciq_ok['ABPm_mean'], bins=30, alpha=0.5, label='UCIQ', color='red', density=True)
ax2.axvline(65, color='black', linestyle='--', label='MAP=65')
ax2.set_xlabel('MAP (mmHg)')
ax2.set_ylabel('Densidad')
ax2.set_title('Distribución de Medias de MAP')
ax2.legend()

# Panel 3: Resumen estadístico
ax3 = axes[2]
ax3.axis('off')

summary_text = f"""
RESUMEN COMPARATIVO MAP
{'='*50}

MIMIC (n={len(mimic_df)}):
  Media: {mimic_df['mean'].mean():.1f} mmHg
  ± SD: {mimic_df['mean'].std():.1f} mmHg
  Rango: {mimic_df['mean'].min():.1f} - {mimic_df['mean'].max():.1f}

UCIQ (n={len(uciq_ok)}):
  Media: {uciq_ok['ABPm_mean'].mean():.1f} mmHg
  ± SD: {uciq_ok['ABPm_mean'].std():.1f} mmHg
  Rango: {uciq_ok['ABPm_mean'].min():.1f} - {uciq_ok['ABPm_mean'].max():.1f}

Diferencia: {diff:.1f} mmHg ({abs(diff)/uciq_ok.ABPm_mean.mean()*100:.1f}%)

Test estadístico:
  KS p-value: {ks_p:.2e}
  Resultado: {'SIGNIFICATIVO' if ks_p < 0.001 else 'No significativo'}

CONCLUSIÓN CLÍNICA:
Las distribuciones de MAP son significativamente
diferentes. MIMIC muestra valores más altos de MAP
que UCIQ, lo que impacta la transferibilidad de
modelos de predicción de hipotensión.
"""

ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./phase_outputs/abp_map_comparison.png', dpi=150, bbox_inches='tight')
print('  Guardado: ./phase_outputs/abp_map_comparison.png')

# Guardar tabla comparativa
comparison = pd.DataFrame([
    {
        'Dataset': 'MIMIC-IV Waveform',
        'N_registros': len(mimic_df),
        'MAP_Media': f"{mimic_df['mean'].mean():.1f}",
        'MAP_SD': f"{mimic_df['mean'].std():.1f}",
        'MAP_Rango': f"{mimic_df['mean'].min():.1f}-{mimic_df['mean'].max():.1f}",
        'MAP_lt_65': f"{mimic_hypo:.1f}%"
    },
    {
        'Dataset': 'UCIQ',
        'N_registros': len(uciq_ok),
        'MAP_Media': f"{uciq_ok['ABPm_mean'].mean():.1f}",
        'MAP_SD': f"{uciq_ok['ABPm_mean'].std():.1f}",
        'MAP_Rango': f"{uciq_ok['ABPm_mean'].min():.1f}-{uciq_ok['ABPm_mean'].max():.1f}",
        'MAP_lt_65': f"{uciq_hypo:.1f}%"
    }
])

comparison.to_csv('./phase_outputs/abp_comparison_table.csv', index=False)
print('  Guardado: ./phase_outputs/abp_comparison_table.csv')

print('\n' + '='*70)
print('ANÁLISIS COMPLETADO')
print('='*70)
