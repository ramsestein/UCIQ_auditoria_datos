import os
import random
import wfdb
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PN_DIR = "mimic3wdb-matched/1.0"
OUTDIR = "./data/mimic4/mimic3wdb_subset"
MAX_GB = 150

# Para pruebas rápidas, limita el scan (None = todos)
SCAN_LIMIT = 160000  # Reducido para que la prueba sea rápida
SEED = 7
SHUFFLE = True
NUM_WORKERS = 4

# Keywords (ajusta si quieres más estricto)
KW_ART = ["ABP", "ART"]          # arterial line (wave)
KW_CO2 = ["CO2", "ETCO2"]        # capnography wave / (a veces) CO2 channel
KW_ICP = ["ICP"]                # intracranial pressure

os.makedirs(OUTDIR, exist_ok=True)

# Shared state for parallel processing
stats_lock = threading.Lock()
downloaded_count = 0
total_size_bytes = 0

def folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def has_any(sig_names, keywords):
    s = " ".join([x.upper() for x in (sig_names or [])])
    return any(k.upper() in s for k in keywords)

def matches_logic(rec_path: str) -> bool:
    # rec_path suele ser 'p00/p000079/3842928_0001'
    rec_path = rec_path.strip("/")
    parts = rec_path.split('/')
    folder = "/".join(parts[:-1])
    rec_name = parts[-1]
    
    # Header remoto (rápido)
    h = wfdb.rdheader(rec_name, pn_dir=f"{PN_DIR}/{folder}")
    strs = h.sig_name or []
    
    has_art = has_any(strs, KW_ART)
    has_co2 = has_any(strs, KW_CO2)
    has_icp = has_any(strs, KW_ICP)

    return (has_art and has_co2) or has_icp

def dl_whole_record(rec_path: str):
    rec_path = rec_path.strip("/")
    parts = rec_path.split('/')
    folder = "/".join(parts[:-1])
    rec_name = parts[-1]
    
    base_pn = PN_DIR.split('/')[0]
    wfdb.dl_database(
        db_dir=f"{base_pn}/{folder}",
        dl_dir=OUTDIR,
        records=[rec_name],
        keep_subdirs=True,
    )

def process_record(rec, max_bytes):
    global downloaded_count
    
    # Pre-check if already exists
    local_hea = os.path.join(OUTDIR, rec + ".hea")
    if os.path.exists(local_hea):
        return None

    try:
        if matches_logic(rec):
            with stats_lock:
                current_size = folder_size_bytes(OUTDIR)
                if current_size >= max_bytes:
                    return "LIMIT_REACHED"
                
            print(f"  [MATCH] {rec} - Descargando...")
            dl_whole_record(rec)
            
            with stats_lock:
                downloaded_count += 1
                return rec
    except Exception as e:
        print(f"  [ERROR] Procesando {rec}: {e}")
    return None

def main():
    random.seed(SEED)
    max_bytes = int(MAX_GB * 1024**3)

    print("Leyendo lista de records…")
    base_folders = wfdb.get_record_list(PN_DIR)
    print(f"Carpetas de pacientes encontradas: {len(base_folders)}")
    
    folders_to_scan = base_folders[:SCAN_LIMIT] if SCAN_LIMIT else base_folders
    print(f"Lanzando escaneo paralelo de carpetas con {NUM_WORKERS} workers...")
    
    def get_sub_records(folder):
        try:
            return [f"{folder}{r}" for r in wfdb.get_record_list(f"{PN_DIR}/{folder}")]
        except Exception as e:
            print(f"  [ERROR] Lstando records en {folder}: {e}")
            return []

    records = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(get_sub_records, f): f for f in folders_to_scan}
        for future in as_completed(futures):
            records.extend(future.result())
            
    print(f"Records totales encontrados: {len(records)}")
    if SHUFFLE:
        random.shuffle(records)

    print(f"Procesando (Match + Download) en paralelo...")
    selected_count = 0
    limit_reached = False

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_record, r, max_bytes): r for r in records}
        for i, future in enumerate(as_completed(futures), 1):
            if limit_reached:
                continue
                
            res = future.result()
            if res == "LIMIT_REACHED":
                print(f"\nStop: alcanzado objetivo ~{MAX_GB} GB")
                limit_reached = True
                # No podemos cancelar futuros ya en marcha fácilmente sin Python 3.9+ cancel()
                # pero dejamos de procesar resultados.
                continue
                
            if res:
                selected_count += 1
            
            if i % 50 == 0:
                with stats_lock:
                    curr_gb = folder_size_bytes(OUTDIR) / (1024**3)
                print(f"Progreso: {i}/{len(records)} | Descargados: {downloaded_count} | Tamaño: {curr_gb:.2f} GB")

    final_size = folder_size_bytes(OUTDIR)
    print(f"\nHecho. Descargados en esta sesión: {downloaded_count}")
    print(f"Tamaño total carpeta: {final_size/1024**3:.2f} GB")
    print(f"Salida: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()