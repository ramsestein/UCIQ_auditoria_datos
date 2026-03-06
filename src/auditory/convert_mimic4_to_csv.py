import os
import pandas as pd
from pathlib import Path

# Dependencias requeridas:
# pip install wfdb pandas

try:
    import wfdb
except ImportError:
    print("Por favor, instala wfdb: pip install wfdb")
    exit(1)

# Rutas de los directorios
BASE_DIR = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
DATA_DIR = BASE_DIR / "data" / "mimic4"
OUTPUT_DIR = BASE_DIR / "data_csv" / "mimic4"

def process_wfdb_record(filepath):
    """Procesa un archivo WFDB (.hea) y lo guarda completo en CSV."""
    out_folder = OUTPUT_DIR
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"{filepath.stem}.csv"
    
    if out_path.exists():
        print(f" -> Ya convertido: {out_path.name}")
        return

    record_name = str(filepath).replace('.hea', '')
    try:
        # wfdb rdrecord parsea el encabezado y el archivo dat correspondiente
        record = wfdb.rdrecord(record_name)
    except Exception as e:
        print(f" -> ERROR leyendo WFDB record {filepath.name}: {e}")
        return

    record_name = str(filepath).replace('.hea', '')
    
    # Solo queremos procesar los archivos que actúan como "Maestros" o registros simples finales.
    # Los segmentos individuales generados (los que tienen `_` en el nombre) los saltamos
    if '_' in filepath.stem:
        return
            
    try:
        # Convertimos a DataFrame de pandas
        # Para probar la conversión de MultiRecord de wfdb a veces requiere un manejo especial,
        # pero to_dataframe() debería poder con ello si los canales son consistentes.
        # Si falla, usamos nuestro propio ensamblador
        try:
            df = record.to_dataframe()
        except Exception as e:
            print(f" -> Falló to_dataframe nativo para {filepath.name}: {e}. Intentando manual...")
            record_manual = build_record_manually(filepath)
            if record_manual is None: return
            df = record_manual.to_dataframe()
        
        # El index devuelto por to_dataframe() es típicamente un pd.Timedelta
        # Lo reseteamos y lo convertimos a segundos
        df = df.reset_index()
        if 'index' in df.columns:
            if pd.api.types.is_timedelta64_dtype(df['index']):
                df['index'] = df['index'].dt.total_seconds()
            df = df.rename(columns={'index': 'Time_s'})
            
        # Si el DataFrame no tiene filas, no lo guardamos
        if len(df) == 0:
            print(f" -> ERROR: DataFrame vacío para {filepath.name}")
            return
            
        # Nos aseguramos de soltar columnas que sean solo NaN enteros si las hay, o no, 
        # para preservar la estructura original.
        df.to_csv(out_path, index=False)
        print(f" -> Convertido a CSV MULTIRECORD UNIFICADO: {out_path.name} ({len(df)} filas)")
        
    except Exception as e:
        print(f"Error procesando dataframe unificado de {filepath.name}: {e}")

def build_record_manually(filepath):
    """
    Función de respaldo: Si el rdrecord falla (ej. un segmento corrupto),
    leemos el layout del MultiRecord pero luego cargamos fragmento a fragmento,
    llenando de NaNs si alguno falla en vez de abortar todo.
    """
    import numpy as np
    record_name = str(filepath).replace('.hea', '')
    
    try:
        # Leer solo the header
        header = wfdb.rdheader(record_name)
        if not isinstance(header, wfdb.MultiRecord):
            # Si no era MultiRecord y falló antes, asumimos irrecuperable
            return None
            
        # Parámetros básicos para concatenar
        sig_names = header.sig_name
        total_len = header.sig_len
        fs = header.fs
        
        print(f"    -> Ensamblando {header.n_seg} segmentos. Duración total esperada: {total_len} muestras")
        
        # Iniciar matriz vacía de NaNs
        final_signal = np.full((total_len, len(sig_names)), np.nan, dtype=np.float32)
        
        current_idx = 0
        for seg_name in header.seg_name:
            if seg_name == '~':
                # Es un hueco, el len del hueco está en header.seg_len para ese segmento
                # El índice del seg_name actual
                idx_in_list = header.seg_name.index(seg_name, header.seg_name.index(seg_name))
                # Buscamos cuántas muestras dura este hueco
                # wfdb guarda los largos de segmento en header.seg_len
                # Como '~' puede aparecer varias veces, llevamos cuenta iterativa.
                pass 
                
        # Método alternativo: iterar ambos, seg_name y seg_len juntos
        for seg_name, s_len in zip(header.seg_name, header.seg_len):
            if seg_name == '~':
                current_idx += s_len
                continue
            
            # Intentar leer el segmento real
            seg_path = filepath.parent / f"{seg_name}.hea"
            try:
                seg_rec = wfdb.rdrecord(str(filepath.parent / seg_name))
                
                # Mapear las columnas del segmento a las del multirecord maestro
                for i, s_name in enumerate(seg_rec.sig_name):
                    if s_name in sig_names:
                        master_idx = sig_names.index(s_name)
                        final_signal[current_idx:current_idx+s_len, master_idx] = seg_rec.p_signal[:, i]
                        
            except Exception as seg_e:
                print(f"       [!] Omitiendo fragmento corrupto {seg_name}: {seg_e}")
                
            current_idx += s_len
            
        # Construimos un objeto Record "fake" simple de wfdb para usar to_dataframe
        merged_rec = wfdb.Record(
            p_signal=final_signal,
            fs=fs,
            sig_name=sig_names,
            sig_len=total_len
        )
        return merged_rec
        
    except Exception as e:
        print(f"    -> Fallo fatal ensamblando manualmente: {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Buscar todos los archivos .hea dentro de data/mimic4/
    hea_files = list(DATA_DIR.rglob("*.hea"))
    print(f"Encontrados {len(hea_files)} archivos WFDB (.hea) en MIMIC-IV.")
    
    for count, filepath in enumerate(hea_files, 1):
        print(f"\n[{count}/{len(hea_files)}] Procesando: {filepath.name}")
        process_wfdb_record(filepath)
            
    print("\n¡Proceso de conversión a CSV finalizado para MIMIC-IV!")

if __name__ == "__main__":
    main()
