import os
import pandas as pd
import numpy as np
import vitaldb
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
INPUT_DIR = BASE_DIR / "data_csv" / "mimic4"
OUTPUT_DIR = BASE_DIR / "data_vital" / "mimic"

def csv_to_vital(csv_path):
    """Convierte un archivo CSV de MIMIC-IV a formato .vital."""
    out_name = csv_path.stem + ".vital"
    out_path = OUTPUT_DIR / out_name
    
    if out_path.exists():
        print(f" -> Ya convertido: {out_name}")
        return

    print(f"Leyendo CSV: {csv_path.name}...")
    try:
        # Cargamos el CSV completo. 
        # Si el archivo es extremadamente grande (>20M filas), esto consumirá mucha RAM (2-4GB).
        df = pd.read_csv(csv_path)
        
        if 'Time_s' not in df.columns:
            print(f" [!] Error: No se encontró la columna 'Time_s' en {csv_path.name}")
            return
            
        # Intentar convertir Time_s a numérico. 
        # Si falla (por ser string de fecha), convertimos a datetime y sacamos segundos relativos.
        try:
            df['Time_s'] = pd.to_numeric(df['Time_s'])
        except (ValueError, TypeError):
            print(f"  Detectado formato de fecha en Time_s para {csv_path.name}. Convirtiendo...")
            df['Time_s'] = pd.to_datetime(df['Time_s'], errors='coerce')
            df = df.dropna(subset=['Time_s'])
            if len(df) > 0:
                # Restar el primer valor para obtener el tiempo relativo en segundos
                df['Time_s'] = (df['Time_s'] - df['Time_s'].iloc[0]).dt.total_seconds()

        # Determinar frecuencia de muestreo aproximada desde el Time_s
        # (dt = T[1] - T[0])
        if len(df) > 1:
            dt = df['Time_s'].iloc[1] - df['Time_s'].iloc[0]
            srate = round(1.0 / dt, 4) if dt > 0 else 0
        else:
            srate = 0
            
        print(f"  Frecuencia detectada: {srate} Hz. Filas: {len(df)}")
        
        vf = vitaldb.VitalFile()
        
        # Procesar cada columna como una pista (track) excepto Time_s
        for col in df.columns:
            if col == 'Time_s':
                continue
                
            # Extraer valores y filtrar NaNs para el formato de vitaldb
            # vitaldb.add_track requiere 'dt' (tiempo relativo) y 'val' (valor)
            vals = df[col].values
            times = df['Time_s'].values
            
            # Filtramos NaNs porque vitaldb suele rechazar nulos en la exportación binaria o desperdiciar espacio
            # Para mantener la sincronía, vitaldb utiliza el 'dt' de cada punto si no es uniforme, 
            # pero aquí lo rellenamos como bloques.
            mask = ~np.isnan(vals)
            valid_vals = vals[mask]
            valid_times = times[mask]
            
            if len(valid_vals) > 0:
                track_name = f"MIMIC/{col}"
                # Empaquetamos en el formato que la API de VitalFile.add_track espera:
                # Una lista de diccionarios {'dt': tiempo, 'val': [valor]}
                # Nota: Usamos float32 para ahorrar espacio en el binario vital.
                recs = [{'dt': float(valid_times[i]), 'val': np.array([valid_vals[i]], dtype=np.float32)} 
                        for i in range(len(valid_vals))]
                
                vf.add_track(track_name, recs, srate=srate)
                print(f"  Pista agregada: {track_name} ({len(valid_vals)} puntos)")
        
        # Guardar el archivo .vital
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        vf.to_vital(str(out_path))
        print(f" -> Guardado con éxito: {out_path.name}")
        
    except Exception as e:
        print(f"Error procesando {csv_path.name}: {e}")

def main():
    print(f"Buscando CSVs en {INPUT_DIR}...")
    if not INPUT_DIR.exists():
        print(f"Error: No existe el directorio de entrada {INPUT_DIR}")
        return
        
    csv_files = list(INPUT_DIR.glob("*.csv"))
    print(f"Encontrados {len(csv_files)} archivos CSV.")
    
    for i, csv_file in enumerate(csv_files, 1):
        # Evitar procesar los segmentos individuales si ya existe el unificado (sin _XXXX)
        # O si el usuario quiere todos los CSVs de la carpeta, los procesamos todos.
        # Por ahora, procesamos todos los que estén en data_csv/mimic4
        print(f"\n[{i}/{len(csv_files)}] Procesando...")
        csv_to_vital(csv_file)

    print("\n¡Conversión de CSV a .vital finalizada!")

if __name__ == "__main__":
    main()
