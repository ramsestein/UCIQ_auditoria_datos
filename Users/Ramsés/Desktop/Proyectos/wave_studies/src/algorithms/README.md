README — `src/algorithms`

Este documento describe en detalle cada módulo de `src/algorithms`, su propósito, entradas esperadas, salida y una breve explicación del método implementado. Los módulos están pensados para recibir como input la representación estándar usada en el proyecto (un objeto `VitalFile` o un `dict` con claves como `signals`, `fs`, `meta`).

Formato general de uso
- Input recomendado: `data` donde `data['signals']` es un dict de arrays, `data['fs']` frecuencia de muestreo, y `data['meta']` metadatos del registro.
- Output: generalmente un `dict` con métricas (scalars, series) y metadatos (`method`, `params`, `timestamp`).

Listado de módulos y descripción técnica

- `baroreflex_sensitivity`:
	- Propósito: estimar la sensibilidad del barorreflejo (BRS) a partir de pares RR-ABP.
	- Entradas: series de presión arterial arterial (ABP) y series de intervalos RR o señal ECG preprocesada.
	- Salida: BRS (ms/mmHg) por método (sequence, transfer-function), y métricas auxiliares (n_seqs, mean_rr, mean_sbp).
	- Método: detección de secuencias (sustituciones lineales RR vs SBP) y/o estimación en frecuencia por TF entre SBP y RR.

- `blood_pressure_variability`:
	- Propósito: cuantificar la variabilidad de la presión arterial en dominios tiempo/frecuencia.
	- Entradas: señal ABP (presión sistólica/diastólica) y `fs`.
	- Salida: SD, CV, potencias LF/HF, índices espectrales.
	- Método: detección de pulsos, extracción de series beat-to-beat, PSD (Welch).

- `cardiac_output`:
	- Propósito: estimación no invasiva del gasto cardíaco a partir de proxy(s) (ej. índice de pulso, presión arterial).
	- Entradas: ABP continuo y parámetros de calibración (si existen).
	- Salida: CO estimado (L/min) y quality flag.
	- Método: modelos empíricos basados en integración de área bajo la curva arterial o regresiones entrenadas en datos de referencia.

- `cardiac_power_output`:
	- Propósito: calcular la potencia cardíaca (CPO = CO * MAP / 451).
	- Entradas: CO y presión arterial media (MAP).
	- Salida: CPO (W) y resumen estadístico.

- `check_availability`:
	- Propósito: utilidades para chequear presencia/longitud/ruido de señales en un `VitalFile`.
	- Entradas: `data['signals']` y umbrales mínimos.
	- Salida: flags booleanos, porcentajes de cobertura por señal.

- `driving_pressure`:
	- Propósito: cálculo de driving pressure (Pplat - PEEP) para ventilación mecánica.
	- Entradas: señales de presión respiratoria (si disponibles) o metadatos ventilador.
	- Salida: driving pressure y distribución por registro.

- `dynamic_compliance`:
	- Propósito: estimar compliance dinámica (Vt / (Ppeak - PEEP)).
	- Entradas: volumen tidal (Vt) o proxy, presiones respiratorias.
	- Salida: compliance dinámica por instante y resumen.

- `effective_arterial_elastance`:
	- Propósito: estimar Ea como relación entre presión y volumen sistólico.
	- Entradas: presión arterial y estimador de volumen sistólico.
	- Salida: Ea (mmHg/mL) y métricas de confianza.

- `heart_rate_variability`:
	- Propósito: calcular métricas tiempo/frecuencia/no lineales de HRV.
	- Entradas: serie de RR (ms) o timestamps de latidos extraídos de ECG.
	- Salida: RMSSD, SDNN, pNN50, espectros LF/HF, entropías.
	- Método: extracción de latidos (Pan-Tompkins via `util_AL`) y análisis estándar (Welch, NN histogram, etc.).

- `icp_model_semi`:
	- Propósito: modelos semi-supervisados para detectar y predecir elevaciones de ICP.
	- Entradas: series de ICP (si disponibles) y señales relacionadas (ABP, CO2, RESP).
	- Salida: probabilidad de hipertensión intracraneal, predicción a corto plazo y logs de confianza.
	- Método: pipeline de feature engineering + modelo semi-supervisado (clustering + clasificador) y validación cruzada.

- `icp_online_monitor`:
	- Propósito: componente de monitorización en tiempo real que consume ventanas y entrega alertas.
	- Entradas: ventanas deslizantes de señal ICP.
	- Salida: alertas, métricas OS (sensitivity/specificity) y estado del modelo.

- `icp_semi_predict`:
	- Propósito: versión de inferencia batch del pipeline de ICP (predicción offline).
	- Entradas: series temporales completas o segmentos etiquetados.
	- Salida: series de predicción y reportes por paciente/registro.

- `respiratory_sinus_arrhythmia`:
	- Propósito: cuantificar RSA (acoplamiento respiración–FC) como biomarcador autonómico.
	- Entradas: series de respiración (RESP) y RR/ECG.
	- Salida: índice RSA, coherencia RESP–RR.
	- Método: análisis de coherencia y métodos basados en filtrado empírico.

- `rox_index`:
	- Propósito: cálculo del índice ROX (SpO2/FiO2 sobre FR) para soporte respiratorio.
	- Entradas: SpO2, FR (o proxy), FiO2 (si disponible)
	- Salida: ROX y etiquetas de riesgo.

- `shock_index`:
	- Propósito: calcular el Shock Index (HR / SBP) y variantes normalizadas.
	- Entradas: HR y PAS instantánea o beat-to-beat.
	- Salida: series de Shock Index y resumen estadístico.

- `systemic_vascular_resistance`:
	- Propósito: estimación de RVS usando MAP, CO y constants.
	- Entradas: MAP, CO, datos demográficos opcionales.
	- Salida: RVS (dyn·s·cm−5) y banderas de calidad.

- `temp_comparison`:
	- Propósito: comparar y validar señales de temperatura entre sensores (core vs periférico).
	- Entradas: series temporales de temperatura.
	- Salida: sesgos, drift, y flags de inconsistencias.

- `util_AL`:
	- Propósito: utilidades comunes (Pan-Tompkins para detección de QRS, filtros, resampling).
	- Entradas: señales crudas; funciones exportadas: `pan_tompkins`, `butter_filter`, `resample_signal`.
	- Salida: latidos detectados, series filtradas.

- `volumetric_capnography`:
	- Propósito: cálculos volumétricos a partir de CO2 espirado (VD/VT, slopes, etc.).
	- Entradas: señal de CO2 y volumen o flujo respiratorio.
	- Salida: parámetros de capnografía volumétrica y calidad de cálculo.

Buenas prácticas y reproducibilidad
- Todos los módulos documentan parámetros configurables vía argumentos o `params` dict.
- Los módulos deben ser idempotentes: ejecutar varias veces sobre la misma entrada produce los mismos outputs.
- Para reproducir resultados, usar el entorno virtual provisto en `requirements.txt` y ejecutar los scripts desde la raíz del proyecto.

Referencias
- Cada módulo incluye referencias en su docstring cuando aplica (artículos, benchmarks, implementaciones de referencia).

Contacto
- Para dudas técnicas o mejora de algoritmos, abrir un issue en el tracker del repositorio o contactar al autor del proyecto.
