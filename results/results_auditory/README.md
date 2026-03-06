README — `results/results_auditory`

Resumen ejecutivo
Este directorio contiene la versión final y corregida de todos los resultados del análisis auditivo/clinical que respaldan el informe clínico principal. Aquí están las tablas, figuras y ficheros de metadatos que deben usarse para generación de reportes y publicación de resultados.

Contenido principal y significado
- `clinical_metadata_audit.csv`: metadata maestro por cada archivo `.vital`. Columnas clave: `filename`, `duration_min`, `track_count`, `tracks` (listado), `box`, `date` y banderas de presencia de señales (ECG, ART, SPO2, CO2, BIS, NEURO, HEMO, VENT, TEMP). Nota: `has_resp` y `has_icp` se derivan a tiempo de ejecución a partir de `tracks`.
- `sampling_rates_summary.csv`: resumen de frecuencias de muestreo por tipo de señal, usado para validar calidad y compatibilidad de algoritmos.
- `session_duration_stats.csv`: distribución de duración de grabaciones (minutos/hours), por box y por mes.
- `artifact_detection_*`: (archivos `artifact_detection_summary.csv`, `artifact_detection_detail.csv`) resultados del detector de artefactos aplicado a series crudas.
- `comparison_table.csv`, `comparison_radar.png`, `comparison_bar_signals.png`: resultados y figuras de la comparación Clinical vs MIMIC-IV Waveform (MIMIC-III numeric removido; ICP incluido). Estos archivos se generaron con el script `compare_mimic_biosignals.py`.
- `mimic_waveform_proportions.csv`: proporciones de presencia de señales en el dataset de MIMIC consultado.

Metodología (paso a paso)
1) Recolección de datos
	- Casos GE CARESCAPE exportados en formato `.vital` desde VitalDB.
	- Metadatos consolidados en `clinical_metadata_audit.csv` mediante `extract_clinical_metadata.py`.

2) Normalización y mapeo de señales
	- Normalizamos nombres de señales a un `signal_map` común (ECG, ART, PLETH, RESP, ICP, CO2, etc.).
	- Registros con múltiples variantes de nombre se agrupan por regex (ej. `\bRR\b|RESP|VENT_RR` → RESP).

3) Control de calidad y detección de artefactos
	- Detección automática de artefactos en ventanas (picos, saturaciones, flatline) usando `analyze_artifacts.py`.
	- Reportes por caja y por señal en `artifact_detection_*`.

4) Análisis de muestreo y duración
	- Agrupación por frecuencia efectiva (`sampling_rates_summary.csv`) y verificación de consistencia temporal.

5) Métricas derivadas y algoritmos
	- Se ejecutaron los algoritmos documentados en `src/algorithms` para generar métricas cardiovasculares, respiratorias y neurológicas.
	- Cada algoritmo escribe outputs normalizados en `results/results_auditory/` y añade metadatos `method`/`params`.

6) Comparación con MIMIC-IV Waveform
	- Se compararon prevalencias de señales y métricas principales entre nuestros datos y MIMIC-IV Waveform (N=200). Se retiró cualquier uso de la antigua porción MIMIC-III numeric.
	- Las banderas `has_resp` y `has_icp` en la comparación se derivan del campo `tracks` por regex (RESP) y (ICP) respectivamente.

7) Síntesis del informe clínico
	- `synthesize_clinical_report.py` agrega secciones a `final_clinical_audit_report.md` usando las tablas y figuras finales.

Reproducibilidad — pasos para regenerar resultados
1. Activar entorno virtual:
```powershell
& .\venv\Scripts\Activate.ps1
```
2. Ejecutar extracción y preprocesado (si hay nuevos `.vital`):
```powershell
python src/auditory/extract_clinical_metadata.py
python src/auditory/analyze_sampling_rates.py
python src/auditory/analyze_artifacts.py
```
3. Generar comparaciones y figuras:
```powershell
python src/auditory/fetch_mimic_waveform_proportions.py
python src/auditory/compare_mimic_biosignals.py
python src/auditory/visualize_clinical_audit.py
```
4. Regenerar informe final:
```powershell
python src/auditory/synthesize_clinical_report.py
```

Versionado y entorno
- Python packages: usar `requirements.txt` en la raíz. Se recomienda un `venv` con Python 3.10+.
- Datos: la carpeta `data/` contiene los `.vital` y subcarpetas `mimic4/`, `vitaldb/`.

Notas y advertencias
- No modifiques manualmente los archivos en `results/results_auditory/` salvo para reproducir; esos ficheros reflejan la versión final aceptada del análisis.
- Si importas o mueves scripts, actualiza rutas base dentro de los scripts (la mayoría usa una variable `BASE`/`RESULTS` que puede requerir ajuste a `results/results_auditory`).

Contacto y trazabilidad
- Los scripts fuente que generaron cada archivo están en [src/auditory](src/auditory) y los algoritmos en [src/algorithms](src/algorithms). Para reproducir un fichero concreto, buscar el script con el mismo nombre o revisar su docstring.
