# UCIQ Audit: Clinical Audit Framework for ICU Biosignal Quality

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Auditoría Comparativa entre MIMIC-IV Waveform (US MICU) y UCIQ (Barcelona SICU)**

Este repositorio contiene una auditoría clínica comparativa entre dos bases de datos de waveform de UCI:

- **MIMIC-IV Waveform**: 200 registros, ~28 horas mediana, ICU médica (Boston, USA)
- **UCIQ**: 1000 registros, ~1.2 horas mediana, ICU quirúrgica (Barcelona, España)

### Argumento Principal (3 Pilares)

1. **MONITORIZAMOS DIFERENTE** (Fenotipos χ²=115, p<0.001)
   - MIMIC: 6.4 señales por registro (patrones heterogéneos)
   - UCIQ: 22.6 señales por registro (patrones estandarizados)

2. **LOS PACIENTES SON DIFERENTES** (Demografía Fase 5B)
   - MIMIC: ICU general mixta, ~65 años
   - UCIQ: UCI específica, ~58 años, European Mediterranean

3. **LOS VALORES HEMODINÁMICOS SON DIFERENTES** (MAP Fase 2B)
   - MIMIC: MAP 76.0 ± 22.1 mmHg, 36.5% hipotensión
   - UCIQ: MAP 79.9 ± 11.9 mmHg, 9.9% hipotensión
   - Diferencia: 3.7× más hipotensión en MIMIC (KS=0.306, p<0.001)

## 📁 Estructura del Repositorio

```
auditoria/
├── docs/                           # Documentación principal
│   ├── UNIFIED_MASTER_DOCUMENT.txt # Documento maestro completo
│   └── CORRECTIONS_SUMMARY.txt     # Resumen de correcciones
│
├── src/                            # Código fuente
│   ├── auditory/                   # Scripts de auditoría clínica
│   │   ├── fix_numerics_extraction.py
│   │   ├── compare_abp_distributions.py
│   │   ├── generate_paper_summary.py
│   │   └── ...
│   └── analysis_clinic/            # Scripts de fases de análisis
│       ├── phase_2b_physiological.py
│       ├── phase_6_phenotype_v2.py
│       ├── phase_6b_transferability.py
│       └── ...
│
├── outputs/                        # Resultados y figuras
│   ├── abp_map_comparison.png      # Comparación MAP
│   ├── paper_summary_table.csv     # Tabla resumen 3 pilares
│   ├── uciq_numerics_summary.csv   # Datos numéricos UCIQ
│   └── mimic_numerics_summary.csv  # Datos numéricos MIMIC
│
├── results/                        # Resultados de auditoría
│   └── results_auditory/           # Resultados detallados
│
└── scripts/                        # Scripts de utilidad
    ├── check_mimic_channels.py
    ├── verify_abp.py
    └── ...
```

## 🔧 Fases Completadas

### Fase 2B: Valores Fisiológicos ✅
- Comparación de distribuciones MAP (Presión Arterial Media)
- 256M+ muestras MIMIC vs 1.1M+ muestras UCIQ
- KS test: p < 0.001 (diferencia highly significant)

### Fase 6: Fenotipos de Monitorización ✅
- 6 fenotipos identificados (Standard, Hemodynamic, Neurological, etc.)
- χ² = 115, p < 0.001 entre datasets

### Fase 6B: Transferibilidad ✅
- AUROC MIMIC→UCIQ: 0.844 [0.828, 0.859]
- AUROC UCIQ→MIMIC: 0.999 [0.998, 1.000]
- Transfer gap: 0.156 (MODERATE domain shift)

## 📈 Resultados Clave

### Comparación de Prevalencia de Señales
| Señal | MIMIC | UCIQ | Diferencia |
|-------|-------|------|------------|
| ECG | 99.5% | 99.6% | +0.1% |
| RESP | 99.5% | 99.3% | -0.2% ✅ |
| ABP (invasiva) | 32.0% | 52.6% | +20.6% |
| ICP | 3.5% | 21.1% | +17.6% |
| CO2 | 0.5% | 24.6% | +24.1% |

### Fenotipos de Monitorización
| Fenotipo | MIMIC | UCIQ |
|----------|-------|------|
| Standard_Monitoring | 66.5% | 47.0% |
| Hemodynamic_Monitoring | 18.0% | 23.0% |
| Neurological_Monitoring | 3.0% | 13.2% |
| Ventilated_Hemodynamic | 0.0% | 16.2% |

## 🚀 Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Ejecutar análisis de MAP
```bash
python src/auditory/quick_abp_comparison.py
```

### Generar resumen del paper
```bash
python src/auditory/generate_paper_summary.py
```

## 📚 Documentos Principales

- **UNIFIED_MASTER_DOCUMENT.txt**: Documento completo con todos los hallazgos
- **CORRECTIONS_SUMMARY.txt**: Resumen de correcciones críticas aplicadas
- **paper_summary_table.csv**: Tabla resumen del argumento de 3 pilares

## 🔬 Hallazgos Críticos

1. **Corrección RESP**: Prevalencia UCIQ corregida de 76.2% a 99.3%
2. **AUROC con Bootstrap**: Intervalos de confianza 95% implementados
3. **MAP Comparison**: Diferencia fisiológica significativa identificada

## 📖 Cita

Si utilizas este análisis, por favor cita:

```
Auditoría Comparativa MIMIC-IV vs UCIQ: Análisis de Transferibilidad
de Modelos de ML en Datos de Waveform de UCI
```

## ⚠️ Limitaciones

- MIMIC: Datos numéricos HR/SpO2/RR no disponibles en archivos de waveform
- Datos demográficos individuales no disponibles para linkage directo

## 🔗 Contacto

Para preguntas sobre este análisis de auditoría, referirse a los documentos
en `docs/` o revisar el código en `src/`.

---

**Fecha**: 2026-04-06  
**Versión**: 0.1
