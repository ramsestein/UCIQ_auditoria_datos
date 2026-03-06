import os
import csv
import re
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
DATA_DIR = BASE_DIR / "data_vital" / "clinic"
OUTPUT_CSV = BASE_DIR / "results_auditory" / "clinical_metadata_audit.csv"

try:
    import vitaldb
except Exception:
    vitaldb = None

SIGNAL_KEYWORDS = {
    'ecg': ['ECG', 'ECG_HR', 'ECG_II', 'ECG_III', 'ECG_VPC_CNT'],
    'art': ['ART', 'ART_MEAN', 'ART_SYS', 'ART_DIA', 'ABP'],
    'co2': ['CO2', 'AWRAY_CO2', 'AWAY_CO2', 'CO2_ET'],
    'spo2': ['SPO2', 'PLETH_SAT_O2', 'PLETH'],
    'bis': ['BIS'],
    'neuro': ['EEG', 'NEURO'],
    'hemo': ['Hemo', 'HGB', 'HEMO'],
    'vent': ['VENT', 'TV_INSP', 'TV_EXP', 'MV_INSP', 'MV_EXP'],
    'temp': ['TEMP', 'T_C', 'TEMP_C']
}

# Helper to parse date from filename like foo_241109_011853.vital -> 2024-11-09
def parse_date_from_filename(fname):
    m = re.search(r"_(\d{6})_", fname)
    if not m:
        return ''
    yymmdd = m.group(1)
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 2000 + yy if yy < 70 else 1900 + yy
    try:
        return f"{year:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return ''


def detect_flags(track_names):
    flags = {k: 0 for k in ['ecg','art','co2','spo2','bis','neuro','hemo','vent','temp']}
    if not track_names:
        return flags
    for tr in track_names:
        for k, keywords in SIGNAL_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in tr.lower():
                    flags[k] = 1
                    break
    return flags


def main():
    if not OUTPUT_CSV.exists():
        print(f"Error: {OUTPUT_CSV} no existe. Ejecuta primero el extractor de metadatos.")
        return

    # Read existing filenames
    existing = set()
    with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing.add(r['filename'])

    # Walk data_vital/clinic
    new_records = []
    for p in DATA_DIR.rglob('*.vital'):
        fname = p.name
        if fname in existing:
            continue
        # Attempt to read minimal metadata
        track_names = []
        duration_min = ''
        track_count = 0
        if vitaldb is not None:
            try:
                vf = vitaldb.VitalFile(str(p))
                track_names = vf.get_track_names()
                track_count = len(track_names)
                # sample first track at 10s interval to estimate duration
                if track_names:
                    try:
                        df = vf.to_pandas(track_names=[track_names[0]], interval=10)
                        if not df.empty:
                            duration_min = round((len(df) * 10) / 60.0, 2)
                    except Exception:
                        duration_min = ''
            except Exception as e:
                print(f"Warning: no se pudo abrir {fname}: {e}")
        else:
            # No vitaldb -> leave blanks
            track_names = []
            track_count = 0
            duration_min = ''

        flags = detect_flags(track_names)
        complexity = sum(flags.values())
        tracks_field = ','.join(track_names)
        box = ''
        # try to infer box from path parts containing 'box'
        for part in p.parts:
            if part.lower().startswith('box'):
                box = part
                break
        date = parse_date_from_filename(fname)

        record = {
            'filename': fname,
            'full_path': str(p),
            'duration_min': duration_min if duration_min != '' else '',
            'track_count': track_count,
            'has_ecg': flags['ecg'],
            'has_art': flags['art'],
            'has_co2': flags['co2'],
            'has_spo2': flags['spo2'],
            'has_bis': flags['bis'],
            'has_neuro': flags['neuro'],
            'has_hemo': flags['hemo'],
            'has_vent': flags['vent'],
            'has_temp': flags['temp'],
            'complexity_score': complexity,
            'tracks': tracks_field,
            'box': box,
            'date': date
        }
        new_records.append(record)

    if not new_records:
        print('No se encontraron nuevos .vital para añadir.')
        return

    # Append to CSV
    fieldnames = ['filename','full_path','duration_min','track_count','has_ecg','has_art','has_co2','has_spo2','has_bis','has_neuro','has_hemo','has_vent','has_temp','complexity_score','tracks','box','date']
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in new_records:
            writer.writerow(r)

    print(f"Añadidos {len(new_records)} nuevos registros a {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
