import wfdb

PN_BASE = "mimic3wdb-matched/1.0"
TEST_REC = "3842928_0001"
TEST_FOLDER = "p00/p000079/"

print(f"--- Prueba 1: rdheader('{TEST_FOLDER}{TEST_REC}', pn_dir='{PN_BASE}') ---")
try:
    h1 = wfdb.rdheader(f"{TEST_FOLDER}{TEST_REC}", pn_dir=PN_BASE)
    print(f"Logrado! Señales: {h1.sig_name}")
except Exception as e:
    print(f"Fallo 1: {e}")

print(f"\n--- Prueba 2: rdheader('{TEST_REC}', pn_dir='{PN_BASE}/{TEST_FOLDER.strip('/')}') ---")
try:
    h2 = wfdb.rdheader(TEST_REC, pn_dir=f"{PN_BASE}/{TEST_FOLDER.strip('/')}")
    print(f"Logrado! Señales: {h2.sig_name}")
except Exception as e:
    print(f"Fallo 2: {e}")
