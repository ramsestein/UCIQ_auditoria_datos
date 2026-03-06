import os
import random
import pandas as pd
import vitaldb

TRKS_URL = "https://api.vitaldb.net/trks"

# --- Ajustes ---
OUTDIR = "./vital_full_cases"
MODE = "AND"          # "OR" = ART o CO2 ; "AND" = ART y CO2
MAX_GB = 100          # objetivo aproximado de tamaño total descargado
SEED = 7
SHUFFLE = True

ART_TNAME = "SNUADC/ART"
CO2_TNAME = "Primus/CO2"

os.makedirs(OUTDIR, exist_ok=True)

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

def main():
    random.seed(SEED)

    print("Descargando track list (trks)…")
    trks = pd.read_csv(TRKS_URL)

    art_cases = set(trks.loc[trks["tname"] == ART_TNAME, "caseid"].dropna().unique().tolist())
    co2_cases = set(trks.loc[trks["tname"] == CO2_TNAME, "caseid"].dropna().unique().tolist())

    if MODE.upper() == "AND":
        target_cases = sorted(list(art_cases & co2_cases))
        print(f"Casos con ART Y CO2: {len(target_cases)}")
    else:
        target_cases = sorted(list(art_cases | co2_cases))
        print(f"Casos con ART O CO2: {len(target_cases)}")

    if SHUFFLE:
        random.shuffle(target_cases)

    # Log reproducible
    pd.Series(target_cases).to_csv(os.path.join(OUTDIR, f"caseids_{MODE}.csv"), index=False, header=False)

    max_bytes = int(MAX_GB * 1024**3)
    downloaded = 0

    for i, caseid in enumerate(target_cases, 1):
        out_path = os.path.join(OUTDIR, f"{int(caseid):04d}.vital")
        if os.path.exists(out_path):
            continue

        current = folder_size_bytes(OUTDIR)
        if current >= max_bytes:
            print(f"\nStop: alcanzado objetivo ~{MAX_GB} GB (actual: {current/1024**3:.2f} GB)")
            break

        print(f"[{i}/{len(target_cases)}] Descargando caseid={caseid} -> {out_path}")

        # Descarga el .vital completo del caso (todas las señales/numerics del archivo)
        vf = vitaldb.VitalFile(int(caseid))
        vf.to_vital(out_path)

        downloaded += 1

    final = folder_size_bytes(OUTDIR)
    print(f"\nHecho. Descargados nuevos: {downloaded}")
    print(f"Tamaño total carpeta: {final/1024**3:.2f} GB")
    print(f"Salida: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()