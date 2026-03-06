#!/usr/bin/env python3
"""
Script para traspasar archivos nuevos de data/clinic a data_vital/clinic
Copia archivos de carpetas *_2 a data_vital/clinic
"""

import os
import shutil
from pathlib import Path

# Rutas base
BASE_DIR = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
SOURCE_DIR = BASE_DIR / "data" / "clinic"
TARGET_DIR = BASE_DIR / "data_vital" / "clinic"

def transfer_new_files():
    """Transfiere archivos de carpetas *_2 a data_vital/clinic"""
    
    # Crear directorio destino si no existe
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    files_skipped = 0
    errors = 0
    
    # Buscar todas las carpetas *_2
    for box_dir in SOURCE_DIR.iterdir():
        if box_dir.is_dir():
            # Buscar carpetas *_2 dentro de cada box
            for subdir in box_dir.iterdir():
                if subdir.is_dir() and subdir.name.endswith('_2'):
                    print(f"Procesando: {subdir}")
                    
                    # Buscar todos los archivos .vital en la carpeta _2
                    for vital_file in subdir.rglob("*.vital"):
                        target_path = TARGET_DIR / vital_file.name
                        
                        if target_path.exists():
                            print(f"  -> Ya existe: {vital_file.name}")
                            files_skipped += 1
                        else:
                            try:
                                shutil.copy2(vital_file, target_path)
                                print(f"  -> Copiado: {vital_file.name}")
                                files_copied += 1
                            except Exception as e:
                                print(f"  -> Error copiando {vital_file.name}: {e}")
                                errors += 1
    
    print(f"\nResumen:")
    print(f"Archivos copiados: {files_copied}")
    print(f"Archivos omitidos (ya existían): {files_skipped}")
    print(f"Errores: {errors}")
    print(f"Total archivos procesados: {files_copied + files_skipped + errors}")

if __name__ == "__main__":
    transfer_new_files()
