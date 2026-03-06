import os
import sys
import vitaldb

# Add src/algorithms to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    import algorithms as Algorithms
    sys.modules['Algorithms'] = Algorithms
    print("Package 'Algorithms' aliased successfully.")
except Exception as e:
    print(f"Failed to alias package: {e}")

# Try to process one file
from add_algorithms_to_vital import process_vital_file

DATA_VITAL_DIR = os.path.join(BASE_DIR, 'data_vital', 'vitaldb')
test_files = [f for f in os.listdir(DATA_VITAL_DIR) if f.endswith('.vital')]

if test_files:
    test_file = os.path.join(DATA_VITAL_DIR, test_files[0])
    print(f"Testing with file: {test_file}")
    
    # Check tracks before
    vf_before = vitaldb.VitalFile(test_file)
    print("Tracks before:", vf_before.get_track_names())
    
    # Process
    process_vital_file(test_file)
    
    # Check tracks after
    vf_after = vitaldb.VitalFile(test_file)
    print("Tracks after:", vf_after.get_track_names())
else:
    print("No vital files found in data_vital/vitaldb")
