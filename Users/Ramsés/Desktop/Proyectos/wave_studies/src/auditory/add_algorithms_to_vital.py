import os
import sys
import pandas as pd
import numpy as np
import vitaldb
import types
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src and src/algorithms to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0] if len(sys.argv) > 0 else __file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
ALGS_DIR = os.path.join(SRC_DIR, 'algorithms')

sys.path.append(SRC_DIR)
try:
    import algorithms
    sys.modules['Algorithms'] = algorithms
except ImportError:
    sys.modules['Algorithms'] = types.ModuleType('Algorithms')

sys.path.append(ALGS_DIR)

from Algorithms.shock_index import ShockIndex
from Algorithms.driving_pressure import DrivingPressure
from Algorithms.dynamic_compliance import DynamicCompliance
from Algorithms.rox_index import RoxIndex
from Algorithms.cardiac_output import CardiacOutput
from Algorithms.systemic_vascular_resistance import SystemicVascularResistance
from Algorithms.cardiac_power_output import CardiacPowerOutput
from Algorithms.effective_arterial_elastance import EffectiveArterialElastance
from Algorithms.heart_rate_variability import HeartRateVariability
from Algorithms.blood_pressure_variability import BloodPressureVariability
from Algorithms.baroreflex_sensitivity import BaroreflexSensitivity
from Algorithms.respiratory_sinus_arrhythmia import RespiratorySinusArrhythmia
from Algorithms.volumetric_capnography import VolumetricCapnography
from Algorithms.util_AL import check_availability

# Mapping of algorithm names to their classes and output track names
ALGORITHMS = {
    'Shock Index': (ShockIndex, 'Derived/Shock_Index', 'SI'),
    'Driving Pressure': (DrivingPressure, 'Derived/Driving_Pressure', 'DP'),
    'Dynamic Compliance': (DynamicCompliance, 'Derived/Dynamic_Compliance', 'DC'),
    'ROX Index': (RoxIndex, 'Derived/ROX_Index', 'RI'),
    'Cardiac Output': (CardiacOutput, 'Derived/Cardiac_Output', 'CO'),
    'Systemic Vascular Resistance': (SystemicVascularResistance, 'Derived/SVR', 'SVR'),
    'Cardiac Power Output': (CardiacPowerOutput, 'Derived/Cardiac_Power_Output', 'CPO'),
    'Effective Arterial Elastance': (EffectiveArterialElastance, 'Derived/Effective_Arterial_Elastance', 'EAE'),
    'Heart Rate Variability': (HeartRateVariability, 'Derived/HRV_SDNN', 'sdnn'),
    'Blood Pressure Variability': (BloodPressureVariability, 'Derived/BPV_STD', 'std'),
    'BRS': (BaroreflexSensitivity, 'Derived/BRS', 'BRS'),
    'RSA': (RespiratorySinusArrhythmia, 'Derived/RSA', 'RSA'),
    'Volumetric Capnography': (VolumetricCapnography, 'Derived/VolCap_VCO2', 'VCO2')
}

def sanitize_vital_file(vf):
    """
    Efficiently finds the real data range and crops the file
    to avoid memory crashes from large gaps (e.g. 0 to Unix Epoch).
    """
    real_starts = []
    real_ends = []
    
    for track_name, trk in vf.trks.items():
        if not trk.recs:
            continue
            
        # Check first and last records to find the range
        # Most of the time, the gap is just a single point at 0 at the start
        first_dt = trk.recs[0].get('dt', 0)
        last_dt = trk.recs[-1].get('dt', 0)
        
        # If the whole track is at 0, ignore it for range calculation
        if last_dt < 1e6:
            continue
            
        # If the first point is at 0 but the last is valid, find the first valid point
        if first_dt < 1e6:
            for r in trk.recs:
                dt = r.get('dt', 0)
                if dt > 1e6:
                    real_starts.append(dt)
                    break
        else:
            real_starts.append(first_dt)
            
        real_ends.append(last_dt)

    if real_starts and real_ends:
        safe_start = min(real_starts)
        safe_end = max(real_ends)
        # Add a small buffer to avoid cropping valid data
        logger.info(f"Cropping file to safe range: {safe_start} - {safe_end}")
        vf.crop(safe_start, safe_end)

def process_vital_file(file_path):
    try:
        vf = vitaldb.VitalFile(file_path)
        
        # Crop the file to actual clinical data range
        sanitize_vital_file(vf)
        
        existing_tracks = vf.get_track_names()
        tracks_to_calculate = check_availability(existing_tracks)
        
        if 'ICP Model' in tracks_to_calculate:
            tracks_to_calculate.remove('ICP Model')
            
        modified = False
        
        for algo_name in tracks_to_calculate:
            if algo_name not in ALGORITHMS:
                continue
                
            algo_cls, track_name, value_col = ALGORITHMS[algo_name]
            
            if track_name in existing_tracks:
                continue
            
            try:
                # instantiate algorithm
                # Use standard __init__ if it doesn't take args, or pass vf to those that do
                instance = algo_cls() if hasattr(algo_cls, '__init__') and (algo_cls.__init__.__code__.co_argcount == 1) else None
                
                if instance and hasattr(instance, 'compute'):
                    results = instance.compute(vf)
                else:
                    results = algo_cls(vf).values
                
                if results is not None and not results.empty:
                    time_col = 'Timestamp' if 'Timestamp' in results.columns else ('Time_ini_ms' if 'Time_ini_ms' in results.columns else results.columns[0])
                    
                    if algo_name == 'Heart Rate Variability':
                        for sub_col, sub_track in [('sdnn', 'Derived/HRV_SDNN'), ('rmsdd', 'Derived/HRV_RMSSD'), ('pnn50', 'Derived/HRV_PNN50')]:
                            if sub_track not in existing_tracks and sub_col in results.columns:
                                data = results[[time_col, sub_col]].dropna()
                                if not data.empty:
                                    recs = [{'dt': row[time_col]/1000.0, 'val': row[sub_col]} for _, row in data.iterrows()]
                                    vf.add_track(dtname=sub_track, recs=recs, srate=0)
                                    modified = True
                    elif algo_name == 'Blood Pressure Variability':
                        for sub_col, sub_track in [('std', 'Derived/BPV_STD'), ('cv', 'Derived/BPV_CV'), ('arv', 'Derived/BPV_ARV')]:
                            if sub_track not in existing_tracks and sub_col in results.columns:
                                data = results[[time_col, sub_col]].dropna()
                                if not data.empty:
                                    recs = [{'dt': row[time_col]/1000.0, 'val': row[sub_col]} for _, row in data.iterrows()]
                                    vf.add_track(dtname=sub_track, recs=recs, srate=0)
                                    modified = True
                    else:
                        data = results[[time_col, value_col]].dropna()
                        if not data.empty:
                            t_sample = data.iloc[0][time_col]
                            t_div = 1000.0 if t_sample > 1e9 else 1.0
                            recs = [{'dt': row[time_col]/t_div, 'val': row[value_col]} for _, row in data.iterrows()]
                            vf.add_track(dtname=track_name, recs=recs, srate=0)
                            modified = True
                    
                    logger.info(f"Added {algo_name} to {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error calculating {algo_name} for {file_path}: {e}")
        
        if modified:
            vf.to_vital(file_path)
            logger.info(f"Saved modified file: {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def main(data_dir):
    vital_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.vital'):
                vital_files.append(os.path.join(root, file))
                
    logger.info(f"Found {len(vital_files)} Vital files to process.")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_vital_file, vital_files)

if __name__ == "__main__":
    DATA_VITAL_DIR = os.path.join(BASE_DIR, 'data_vital')
    main(DATA_VITAL_DIR)
