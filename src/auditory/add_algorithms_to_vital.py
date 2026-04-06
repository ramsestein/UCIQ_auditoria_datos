import os
import sys
import argparse
import pandas as pd
import numpy as np
import vitaldb
import types
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional
import logging

# Set up logging (level configured later via CLI)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Repo root and paths
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
ALGS_DIR = SRC_DIR / 'algorithms'

sys.path.append(str(SRC_DIR))
try:
    import algorithms
    sys.modules['Algorithms'] = algorithms
except ImportError:
    sys.modules['Algorithms'] = types.ModuleType('Algorithms')

sys.path.append(str(ALGS_DIR))

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

# Autonomic subset used for experiment 01
AUTONOMIC_ALGOS = {'Heart Rate Variability', 'Blood Pressure Variability', 'BRS', 'RSA'}

DERIVED_PREFIX = 'Derived/'


def _infer_vf_time_unit(vf: vitaldb.VitalFile) -> str:
    """Infer whether vf dt is in seconds or milliseconds."""
    dts = []
    for name, trk in vf.trks.items():
        if name.startswith(DERIVED_PREFIX):
            continue
        if not trk.recs:
            continue
        dt = trk.recs[-1].get('dt', 0)
        if dt:
            dts.append(float(dt))
    if not dts:
        return 's'
    med = float(np.median(dts))
    return 'ms' if med > 1e11 else 's'


def _to_vf_dt(ts: float, vf_unit: str) -> float:
    """Convert a timestamp (seconds or ms) to match vf dt unit."""
    t = float(ts)
    src_unit = 'ms' if abs(t) > 1e11 else 's'
    if src_unit == vf_unit:
        return t
    if src_unit == 'ms' and vf_unit == 's':
        return t / 1000.0
    if src_unit == 's' and vf_unit == 'ms':
        return t * 1000.0
    return t


def _track_dt_median(vf: vitaldb.VitalFile, dtname: str) -> Optional[float]:
    trk = vf.trks.get(dtname)
    if trk is None or not trk.recs:
        return None
    first_dt = trk.recs[0].get('dt', None)
    last_dt = trk.recs[-1].get('dt', None)
    if first_dt is None or last_dt is None:
        return None
    return float(first_dt + last_dt) / 2.0


def _timebase_mismatch(track_dt: float, ref_dt: float) -> bool:
    if track_dt <= 0 or ref_dt <= 0:
        return False
    ratio = max(track_dt, ref_dt) / max(1e-9, min(track_dt, ref_dt))
    return ratio > 100


def parse_args():
    parser = argparse.ArgumentParser(description='Attach derived algorithm tracks to .vital files.')
    parser.add_argument('--input', default=str(ROOT / 'data_vital' / 'clinic'),
                        help='Directory with .vital files (default: data_vital/clinic)')
    parser.add_argument('--files', default='',
                        help='Optional comma-separated list of .vital filenames (within --input) to process')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Process at most N files (0 = all). Applied after --files filtering if provided.')
    parser.add_argument('--workers', type=int, default=4, help='Thread workers for parallel processing')
    parser.add_argument('--only-autonomic', action='store_true',
                        help='Compute only autonomic metrics (HRV, BPV, BRS, RSA)')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation by deleting existing Derived/* output tracks for selected algorithms')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    return parser.parse_args()

def sanitize_vital_file(vf, max_span_sec=6 * 3600):
    """
    Finds the real data range and crops the file, capping span to avoid
    enormous resampling windows that blow memory.
    """
    real_starts = []
    real_ends = []

    for track_name, trk in vf.trks.items():
        # Ignore derived tracks (they can be in a different/incorrect timebase)
        if track_name.startswith(DERIVED_PREFIX):
            continue
        if not trk.recs:
            continue

        first_dt = trk.recs[0].get('dt', 0)
        last_dt = trk.recs[-1].get('dt', 0)

        if last_dt < 1e6:
            continue

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

        if safe_end - safe_start > max_span_sec:
            safe_end = safe_start + max_span_sec
            logger.info(f"Cropping file to capped range: {safe_start} - {safe_end} (cap {max_span_sec}s)")
        else:
            logger.info(f"Cropping file to safe range: {safe_start} - {safe_end}")

        vf.crop(safe_start, safe_end)

def process_vital_file(file_path, allowed_algos=None, force: bool = False):
    try:
        vf = vitaldb.VitalFile(file_path)
        
        # Crop the file to actual clinical data range
        sanitize_vital_file(vf)
        
        vf_unit = _infer_vf_time_unit(vf)
        existing_tracks = set(vf.get_track_names())
        tracks_to_calculate = check_availability(existing_tracks)

        if allowed_algos:
            tracks_to_calculate = [t for t in tracks_to_calculate if t in allowed_algos]
        
        if 'ICP Model' in tracks_to_calculate:
            tracks_to_calculate.remove('ICP Model')
            
        modified = False

        # Reference dt for detecting corrupted Derived/* tracks
        ref_dts = []
        for name, trk in vf.trks.items():
            if name.startswith(DERIVED_PREFIX) or not trk.recs:
                continue
            ref_dts.append(float(trk.recs[-1].get('dt', 0)))
        ref_dt = float(np.median(ref_dts)) if ref_dts else 0.0
        
        for algo_name in tracks_to_calculate:
            if algo_name not in ALGORITHMS:
                continue
                
            algo_cls, track_name, value_col = ALGORITHMS[algo_name]

            # Multi-output algorithms
            if algo_name == 'Heart Rate Variability':
                out_tracks = ['Derived/HRV_SDNN', 'Derived/HRV_RMSSD', 'Derived/HRV_PNN50']
            elif algo_name == 'Blood Pressure Variability':
                out_tracks = ['Derived/BPV_STD', 'Derived/BPV_CV', 'Derived/BPV_ARV']
            else:
                out_tracks = [track_name]

            if force:
                for out_t in out_tracks:
                    if out_t in existing_tracks:
                        try:
                            vf.del_track(out_t)
                            existing_tracks.discard(out_t)
                            modified = True
                            logger.warning(f"Deleted {out_t} due to --force")
                        except Exception as e:
                            logger.warning(f"Could not delete {out_t} for --force: {e}")

            # Repair obvious timebase mismatches on existing Derived/* tracks
            need_compute = False
            for out_t in out_tracks:
                # If the derived track exists but is empty, delete to allow recompute
                if out_t in existing_tracks:
                    trk_obj = vf.trks.get(out_t)
                    if trk_obj is not None and (not getattr(trk_obj, 'recs', None) or len(trk_obj.recs) == 0):
                        try:
                            vf.del_track(out_t)
                            existing_tracks.discard(out_t)
                            modified = True
                            need_compute = True
                            logger.warning(f"Deleted {out_t} because it existed but had 0 records.")
                        except Exception as e:
                            logger.warning(f"Could not delete empty {out_t}: {e}")

                if out_t in existing_tracks and ref_dt > 0:
                    tmed = _track_dt_median(vf, out_t)
                    if tmed is not None and _timebase_mismatch(tmed, ref_dt):
                        try:
                            vf.del_track(out_t)
                            existing_tracks.discard(out_t)
                            modified = True
                            need_compute = True
                            logger.warning(f"Deleted {out_t} due to timebase mismatch (dt~{tmed} vs ref~{ref_dt}).")
                        except Exception as e:
                            logger.warning(f"Could not delete {out_t}: {e}")

            if not need_compute and all(t in existing_tracks for t in out_tracks):
                continue
            
            try:
                # instantiate algorithm
                # Use standard __init__ if it doesn't take args, or pass vf to those that do
                instance = algo_cls() if hasattr(algo_cls, '__init__') and (algo_cls.__init__.__code__.co_argcount == 1) else None
                
                if instance and hasattr(instance, 'compute'):
                    results = instance.compute(vf)
                else:
                    results = algo_cls(vf).values
                
                if results is None or results.empty:
                    continue

                time_col = 'Timestamp' if 'Timestamp' in results.columns else ('Time_ini_ms' if 'Time_ini_ms' in results.columns else results.columns[0])
                
                if algo_name == 'Heart Rate Variability':
                    for sub_col, sub_track in [('sdnn', 'Derived/HRV_SDNN'), ('rmsdd', 'Derived/HRV_RMSSD'), ('pnn50', 'Derived/HRV_PNN50')]:
                        if sub_track not in existing_tracks and sub_col in results.columns:
                            data = results[[time_col, sub_col]].dropna()
                            if not data.empty:
                                ts = data[time_col].to_numpy(dtype=float)
                                vals = data[sub_col].to_numpy(dtype=float)
                                recs = [{'dt': _to_vf_dt(t, vf_unit), 'val': float(v)} for t, v in zip(ts, vals)]
                                vf.add_track(dtname=sub_track, recs=recs, srate=0)
                                modified = True
                                existing_tracks.add(sub_track)
                elif algo_name == 'Blood Pressure Variability':
                    for sub_col, sub_track in [('std', 'Derived/BPV_STD'), ('cv', 'Derived/BPV_CV'), ('arv', 'Derived/BPV_ARV')]:
                        if sub_track not in existing_tracks and sub_col in results.columns:
                            data = results[[time_col, sub_col]].dropna()
                            if not data.empty:
                                ts = data[time_col].to_numpy(dtype=float)
                                vals = data[sub_col].to_numpy(dtype=float)
                                recs = [{'dt': _to_vf_dt(t, vf_unit), 'val': float(v)} for t, v in zip(ts, vals)]
                                vf.add_track(dtname=sub_track, recs=recs, srate=0)
                                modified = True
                                existing_tracks.add(sub_track)
                else:
                    data = results[[time_col, value_col]].dropna()
                    if not data.empty:
                        ts = data[time_col].to_numpy(dtype=float)
                        vals = data[value_col].to_numpy(dtype=float)
                        recs = [{'dt': _to_vf_dt(t, vf_unit), 'val': float(v)} for t, v in zip(ts, vals)]
                        vf.add_track(dtname=track_name, recs=recs, srate=0)
                        modified = True
                        existing_tracks.add(track_name)
                
                logger.info(f"Added {algo_name} to {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error calculating {algo_name} for {file_path}: {e}")
        
        if modified:
            vf.to_vital(file_path)
            logger.info(f"Saved modified file: {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    data_dir = Path(args.input)
    vital_paths = sorted(data_dir.rglob('*.vital'))

    if args.files:
        wanted = {x.strip() for x in args.files.split(',') if x.strip()}
        vital_paths = [p for p in vital_paths if p.name in wanted]

    if args.max_files and args.max_files > 0:
        vital_paths = vital_paths[:args.max_files]

    vital_files = [str(p) for p in vital_paths]

    allowed = AUTONOMIC_ALGOS if args.only_autonomic else None

    logger.info(f"Found {len(vital_files)} Vital files to process in {data_dir}.")

    worker_fn = partial(process_vital_file, allowed_algos=allowed, force=args.force)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        executor.map(worker_fn, vital_files)


if __name__ == "__main__":
    main()
