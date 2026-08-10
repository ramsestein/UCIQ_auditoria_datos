# UCIQ Audit: Clinical Audit Framework for ICU Biosignal Quality

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Comparative audit of MIMIC-IV Waveform (US MICU) and UCIQ (Barcelona SICU)**

This repository contains a comparative clinical audit of two ICU waveform databases:

- **MIMIC-IV Waveform**: 200 records, ~28 h median duration, medical ICU (Boston, USA)
- **UCIQ**: 1,000 records, ~1.2 h median duration, surgical ICU (Barcelona, Spain)

### Main argument (three pillars)

1. **WE MONITOR DIFFERENTLY** (phenotypes, χ² = 115, p < 0.001)
   - MIMIC: 6.4 signals per record (heterogeneous patterns)
   - UCIQ: 22.6 signals per record (standardised patterns)

2. **THE PATIENTS ARE DIFFERENT** (demographics, Phase 5B)
   - MIMIC: mixed general ICU, ~65 years
   - UCIQ: specialty-specific ICU, ~58 years, Southern European / Mediterranean

3. **THE HAEMODYNAMIC VALUES ARE DIFFERENT** (MAP, Phase 2B)
   - MIMIC: MAP 76.0 ± 22.1 mmHg, 36.5% hypotension
   - UCIQ: MAP 79.9 ± 11.9 mmHg, 9.9% hypotension
   - Difference: 3.7× more hypotension in MIMIC (KS = 0.306, p < 0.001)

## 📁 Repository structure

```
auditoria/
├── docs/                           # Main documentation
│   ├── UNIFIED_MASTER_DOCUMENT.txt # Full master document
│   └── CORRECTIONS_SUMMARY.txt     # Summary of corrections
│
├── src/                            # Source code
│   ├── auditory/                   # Clinical audit scripts
│   │   ├── fix_numerics_extraction.py
│   │   ├── compare_abp_distributions.py
│   │   ├── generate_paper_summary.py
│   │   └── ...
│   └── analysis_clinic/            # Analysis-phase scripts
│       ├── phase_2b_physiological.py
│       ├── phase_6_phenotype_v2.py
│       ├── phase_6b_transferability.py
│       └── ...
│
├── outputs/                        # Results and figures
│   ├── abp_map_comparison.png      # MAP comparison
│   ├── paper_summary_table.csv     # Three-pillar summary table
│   ├── uciq_numerics_summary.csv   # UCIQ numeric data
│   └── mimic_numerics_summary.csv  # MIMIC numeric data
│
├── results/                        # Audit results
│   └── results_auditory/           # Detailed results
│
└── scripts/                        # Utility scripts
    ├── check_mimic_channels.py
    ├── verify_abp.py
    └── ...
```

## 🔧 Completed phases

### Phase 2B: Physiological values ✅
- Comparison of MAP (mean arterial pressure) distributions
- 256M+ MIMIC samples vs 1.1M+ UCIQ samples
- KS test: p < 0.001 (highly significant difference)

### Phase 6: Monitoring phenotypes ✅
- 6 phenotypes identified (Standard, Hemodynamic, Neurological, etc.)
- χ² = 115, p < 0.001 between datasets

### Phase 6B: Transferability ✅
- AUROC MIMIC→UCIQ: 0.844 [0.828, 0.859]
- AUROC UCIQ→MIMIC: 0.999 [0.998, 1.000]
- Transfer gap: 0.156 (moderate domain shift)

## 📈 Key results

### Signal prevalence comparison
| Signal | MIMIC | UCIQ | Difference |
|--------|-------|------|------------|
| ECG | 99.5% | 99.6% | +0.1% |
| RESP | 99.5% | 99.3% | -0.2% ✅ |
| ABP (invasive) | 32.0% | 52.6% | +20.6% |
| ICP | 3.5% | 21.1% | +17.6% |
| CO2 | 0.5% | 24.6% | +24.1% |

### Monitoring phenotypes
| Phenotype | MIMIC | UCIQ |
|-----------|-------|------|
| Standard_Monitoring | 66.5% | 47.0% |
| Hemodynamic_Monitoring | 18.0% | 23.0% |
| Neurological_Monitoring | 3.0% | 13.2% |
| Ventilated_Hemodynamic | 0.0% | 16.2% |

## 🚀 Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Run the MAP analysis
```bash
python src/auditory/quick_abp_comparison.py
```

### Generate the paper summary
```bash
python src/auditory/generate_paper_summary.py
```

## 📚 Key documents

- **UNIFIED_MASTER_DOCUMENT.txt**: Complete document with all findings
- **CORRECTIONS_SUMMARY.txt**: Summary of the critical corrections applied
- **paper_summary_table.csv**: Summary table for the three-pillar argument

## 🔬 Critical findings

1. **RESP correction**: UCIQ prevalence corrected from 76.2% to 99.3%
2. **Bootstrapped AUROC**: 95% confidence intervals implemented
3. **MAP comparison**: significant physiological difference identified

## 📖 Citation

If you use this analysis, please cite:

```
Comparative audit of MIMIC-IV vs UCIQ: transferability analysis
of ML models on ICU waveform data
```

## ⚠️ Limitations

- MIMIC: HR/SpO2/RR numeric data are not available in the waveform files
- Individual-level demographic data are not available for direct linkage

## 🔗 Contact

For questions about this audit analysis, see the documents in `docs/`
or review the code in `src/`.

---

**Date**: 2026-04-06  
**Version**: 0.1
