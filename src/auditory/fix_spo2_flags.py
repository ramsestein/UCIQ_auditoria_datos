import re
import csv
from pathlib import Path

BASE = Path(r"c:\Users\Ramsés\Desktop\Proyectos\wave_studies")
CSV_PATH = BASE / "results_auditory" / "clinical_metadata_audit.csv"

try:
    import vitaldb
except Exception:
    vitaldb = None

# Patterns to consider as SpO2 indicators
SPO2_PATTERNS = [r'spo2', r'pleth', r'sat', r'sat_o2', r'pleth_sat', r'pleth_s']

def tracks_has_spo2(tracks_str):
    if not tracks_str or tracks_str.strip() == '':
        return False
    s = tracks_str.lower()
    for p in SPO2_PATTERNS:
        if re.search(p, s):
            return True
    return False


def check_with_vitalfile(path):
    if vitaldb is None:
        return False
    try:
        vf = vitaldb.VitalFile(str(path))
        names = vf.get_track_names()
        for n in names:
            if tracks_has_spo2(n):
                return True
    except Exception:
        return False
    return False


def main():
    if not CSV_PATH.exists():
        print('CSV not found:', CSV_PATH)
        return

    rows = []
    updated = 0
    total = 0

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            total += 1
            has_spo2 = int(r.get('has_spo2', '0') if r.get('has_spo2','')!='' else 0)
            if has_spo2 == 1:
                rows.append(r)
                continue

            # First try using tracks column
            tracks = r.get('tracks','')
            if tracks_has_spo2(tracks):
                r['has_spo2'] = '1'
                updated += 1
                rows.append(r)
                continue

            # Otherwise try opening vital file
            full_path = r.get('full_path','')
            if full_path:
                if check_with_vitalfile(full_path):
                    r['has_spo2'] = '1'
                    updated += 1
            rows.append(r)

    # Write back CSV (overwrite)
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Procesados: {total}, flags corregidos: {updated}')

if __name__ == '__main__':
    main()
