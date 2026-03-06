import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import vitaldb

def check_file(path):
    print(f"Checking: {path}")
    try:
        vf = vitaldb.VitalFile(path)
        tracks = vf.get_track_names()
        start = float('inf')
        end = float('-inf')
        found = False
        for tr in tracks[:5]: # Check first 5 tracks
            df = vf.to_pandas(tr, return_timestamp=True)
            real_indices = df.index[df.index > 1e6]
            if len(real_indices) > 0:
                start = min(start, real_indices.min())
                end = max(end, real_indices.max())
                found = True
        if found:
            print(f"Real Range: {start} to {end}")
            print(f"Duration: {end - start} seconds")
        else:
            print("No data found > 1,000,000 epoch")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_file(r'C:\Users\Ramsés\Desktop\Proyectos\wave_studies\data_vital\clinic\7gibrwfgp_241109_011853.vital')
