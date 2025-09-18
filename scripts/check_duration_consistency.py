#!/usr/bin/env python3
"""
Check duration consistency between processed robot motions and raw SMPL-X motions.

For each processed npz under `--processed_dir` (default: /home/axell/Desktop/dataset_new/retarget_g1/AMASS),
find the corresponding raw SMPL-X npz under `--raw_dir` (default: /home/axell/Desktop/dataset_new/raw/AMASS)
using the same relative path, then compare durations computed as (T-1)/fps.

Reports files whose duration difference exceeds a threshold (default: 0.2s).

Usage:
  python scripts/check_duration_consistency.py \
      --processed_dir /home/axell/Desktop/dataset_new/retarget_g1/AMASS \
      --raw_dir /home/axell/Desktop/dataset_new/raw/AMASS \
      --threshold 0.2
"""

import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def to_float_scalar(x):
    """Safely convert np array/scalar to Python float."""
    try:
        arr = np.array(x)
        # squeeze down to 0-D and get scalar
        return float(arr.reshape(()))
    except Exception:
        return float(x)


def get_raw_fps(raw_dict):
    """Heuristically find FPS from raw SMPL-X dict."""
    candidates = [
        "mocap_frame_rate",
        "mocap_framerate",
        "frame_rate",
        "framerate",
        "fps",
    ]
    for key in candidates:
        if key in raw_dict:
            try:
                return to_float_scalar(raw_dict[key])
            except Exception:
                pass
    raise KeyError(f"FPS key not found in raw npz (tried {candidates})")


def get_raw_T(raw_dict):
    """Heuristically get frame count T from raw SMPL-X dict."""
    # Prefer pose_body if available
    for key in ["pose_body", "trans", "root_orient", "global_orient"]:
        if key in raw_dict:
            shape = raw_dict[key].shape
            if len(shape) >= 1:
                return int(shape[0])
    # Fallback: search any array with first dimension > 1
    for key, val in raw_dict.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] > 1:
            return int(val.shape[0])
    raise KeyError("Unable to determine frame count T from raw npz")


def get_proc_fps(proc_dict):
    if "fps" not in proc_dict:
        raise KeyError("Processed npz missing 'fps'")
    return to_float_scalar(proc_dict["fps"])


def get_proc_T(proc_dict):
    # Prefer root_pos if available
    for key in ["root_pos", "dof_pos", "root_rot"]:
        if key in proc_dict:
            shape = proc_dict[key].shape
            if len(shape) >= 1:
                return int(shape[0])
    # Fallback: search any array with first dimension > 1
    for key, val in proc_dict.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] > 1:
            return int(val.shape[0])
    raise KeyError("Unable to determine frame count T from processed npz")


def compute_duration(num_frames: int, fps: float) -> float:
    if num_frames <= 1 or fps <= 0:
        return 0.0
    return (num_frames - 1) / float(fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed_dir",
        type=str,
        default="/home/axell/Desktop/dataset_new/retarget_g1/AMASS",
        help="Root folder of processed/retargeted npz files",
    )
    ap.add_argument(
        "--raw_dir",
        type=str,
        default="/home/axell/Desktop/dataset_new/raw/AMASS",
        help="Root folder of raw SMPL-X npz files",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Allowed absolute difference in seconds",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for every file",
    )
    ap.add_argument(
        "--list_mismatches",
        type=str,
        default=None,
        help="Optional path to save a list of mismatched files",
    )
    args = ap.parse_args()

    proc_root = Path(args.processed_dir)
    raw_root = Path(args.raw_dir)

    if not proc_root.exists():
        raise FileNotFoundError(f"Processed dir not found: {proc_root}")
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_root}")

    processed_files = [p for p in proc_root.rglob("*.npz") if p.is_file()]
    processed_files.sort()

    total = 0
    ok = 0
    mismatched = 0
    missing_raw = 0
    errors = 0

    mismatches = []

    for proc_path in tqdm(processed_files, desc="Checking durations"):
        total += 1
        rel = proc_path.relative_to(proc_root)
        raw_path = raw_root / rel

        if not raw_path.exists():
            missing_raw += 1
            if args.verbose:
                print(f"MISSING RAW: {rel}")
            continue

        try:
            proc_npz = np.load(proc_path, allow_pickle=True)
            raw_npz = np.load(raw_path, allow_pickle=True)

            proc_fps = get_proc_fps(proc_npz)
            proc_T = get_proc_T(proc_npz)
            raw_fps = get_raw_fps(raw_npz)
            raw_T = get_raw_T(raw_npz)

            dur_proc = compute_duration(proc_T, proc_fps)
            dur_raw = compute_duration(raw_T, raw_fps)
            diff = abs(dur_proc - dur_raw)

            if diff <= args.threshold:
                ok += 1
                if args.verbose:
                    print(
                        f"OK: {rel} | raw: T={raw_T}, fps={raw_fps:.3f}, dur={dur_raw:.3f}s | "
                        f"proc: T={proc_T}, fps={proc_fps:.3f}, dur={dur_proc:.3f}s | diff={diff:.3f}s"
                    )
            else:
                mismatched += 1
                msg = (
                    f"MISMATCH: {rel} | raw: T={raw_T}, fps={raw_fps:.3f}, dur={dur_raw:.3f}s | "
                    f"proc: T={proc_T}, fps={proc_fps:.3f}, dur={dur_proc:.3f}s | diff={diff:.3f}s"
                )
                mismatches.append(msg)
                print(msg)

        except Exception as e:
            errors += 1
            print(f"ERROR: {rel} | {e}")
            continue

    print("\nSummary")
    print(f"  Total processed files:   {total}")
    print(f"  OK within threshold:     {ok}")
    print(f"  Mismatched (> {args.threshold:.3f}s): {mismatched}")
    print(f"  Missing raw files:       {missing_raw}")
    print(f"  Errors reading files:    {errors}")

    if args.list_mismatches and mismatches:
        out_path = Path(args.list_mismatches)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in mismatches:
                f.write(line + "\n")
        print(f"Saved mismatches to: {out_path}")


if __name__ == "__main__":
    main()

