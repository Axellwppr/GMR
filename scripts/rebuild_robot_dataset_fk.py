#!/usr/bin/env python3

import argparse
import os
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT


DEFAULT_SRC_FOLDER = pathlib.Path(
    "/home/axell/Desktop/dataset_new/retarget_g1/real_vr_data"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild robot dataset FK fields with optional anomaly trimming. "
            "The script uses a fixed max-delta threshold on dof_pos to cut out "
            "abrupt frame jumps, splits the remaining motion into contiguous "
            "segments, then recomputes local_body_pos/local_body_rot via FK "
            "and overwrites body_names/joint_names."
        )
    )
    parser.add_argument(
        "--src-folder",
        type=pathlib.Path,
        default=DEFAULT_SRC_FOLDER,
        help="Folder containing robot motion .npz files. Processed recursively.",
    )
    parser.add_argument(
        "--dst-folder",
        type=pathlib.Path,
        default=None,
        help=(
            "Folder to save converted files. Defaults to a sibling folder named "
            "<src>_segmented."
        ),
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        choices=sorted(ROBOT_XML_DICT.keys()),
        help="Robot model used to rebuild FK fields.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device, e.g. "auto", "cpu", or "cuda:0".',
    )
    parser.add_argument(
        "--no-height-adjust",
        action="store_true",
        help="Disable root height correction.",
    )
    parser.add_argument(
        "--no-root-origin-offset",
        action="store_true",
        help="Disable XY offset that moves the first root frame to the origin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute, print stats, and validate outputs without writing files.",
    )
    parser.add_argument(
        "--no-anomaly-cut",
        action="store_true",
        help="Disable anomaly detection and frame cutting.",
    )
    parser.add_argument(
        "--anomaly-margin",
        type=int,
        default=25,
        help="Remove this many frames before and after each detected anomaly.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.6,
        help="Fixed threshold on max per-frame dof delta.",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=30,
        help="Drop segments shorter than this many frames after trimming.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def list_npz_files(src_folder: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in src_folder.rglob("*.npz") if path.is_file())


def validate_motion_array(name: str, array: np.ndarray, expected_dim: int, expected_len: int):
    if array.ndim != expected_dim:
        raise ValueError(f"{name} ndim mismatch: got {array.ndim}, expected {expected_dim}")
    if array.shape[0] != expected_len:
        raise ValueError(
            f"{name} length mismatch: got {array.shape[0]}, expected {expected_len}"
        )


def compute_max_deltas(dof_pos: np.ndarray) -> np.ndarray:
    if dof_pos.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float64)
    return np.abs(np.diff(np.asarray(dof_pos, dtype=np.float64), axis=0)).max(axis=1)


def summarize_deltas(max_delta: np.ndarray) -> dict[str, float]:
    if max_delta.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "mad": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "p99": 0.0,
            "p99_5": 0.0,
            "p99_9": 0.0,
        }

    median = float(np.median(max_delta))
    q1 = float(np.percentile(max_delta, 25))
    q3 = float(np.percentile(max_delta, 75))
    return {
        "mean": float(np.mean(max_delta)),
        "std": float(np.std(max_delta)),
        "median": median,
        "mad": float(np.median(np.abs(max_delta - median))),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p99": float(np.percentile(max_delta, 99)),
        "p99_5": float(np.percentile(max_delta, 99.5)),
        "p99_9": float(np.percentile(max_delta, 99.9)),
    }


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_anomaly_spans(
    num_frames: int,
    max_delta: np.ndarray,
    *,
    threshold: float,
    anomaly_margin: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    anomaly_frame_idx = np.where(max_delta > threshold)[0] + 1
    spans = []
    for frame_idx in anomaly_frame_idx:
        start = max(0, int(frame_idx) - anomaly_margin)
        end = min(num_frames, int(frame_idx) + anomaly_margin + 1)
        spans.append((start, end))
    return anomaly_frame_idx, merge_spans(spans)


def build_keep_mask(num_frames: int, spans: list[tuple[int, int]]) -> np.ndarray:
    keep_mask = np.ones((num_frames,), dtype=bool)
    for start, end in spans:
        keep_mask[start:end] = False
    return keep_mask


def build_keep_spans(
    num_frames: int,
    cut_spans: list[tuple[int, int]],
    *,
    min_segment_frames: int,
) -> list[tuple[int, int]]:
    if not cut_spans:
        return [(0, num_frames)] if num_frames >= min_segment_frames else []

    keep_spans = []
    cursor = 0
    for start, end in cut_spans:
        if start > cursor and start - cursor >= min_segment_frames:
            keep_spans.append((cursor, start))
        cursor = max(cursor, end)
    if num_frames > cursor and num_frames - cursor >= min_segment_frames:
        keep_spans.append((cursor, num_frames))
    return keep_spans


def slice_framewise_fields(motion_data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    num_frames = int(np.asarray(motion_data["root_pos"]).shape[0])
    sliced = {}
    for key, value in motion_data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == num_frames:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def rebuild_motion_fields(
    kinematics_model: KinematicsModel,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    *,
    height_adjust: bool,
    root_origin_offset: bool,
):
    num_frames = int(root_pos.shape[0])
    device = kinematics_model._device

    root_pos_out = np.array(root_pos, copy=True)
    root_rot_in = np.asarray(root_rot)
    dof_pos_in = np.asarray(dof_pos)

    dof_tensor = torch.from_numpy(dof_pos_in).to(device=device, dtype=torch.float32)

    fk_root_pos = torch.zeros((num_frames, 3), device=device, dtype=torch.float32)
    fk_root_rot = torch.zeros((num_frames, 4), device=device, dtype=torch.float32)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, local_body_rot = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        dof_tensor,
    )

    if height_adjust:
        body_pos, _ = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos_out).to(device=device, dtype=torch.float32),
            torch.from_numpy(root_rot_in).to(device=device, dtype=torch.float32),
            dof_tensor,
        )
        lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos_out[:, 2] = root_pos_out[:, 2] - lowest_height

    if root_origin_offset and num_frames > 0:
        root_pos_out[:, :2] -= root_pos_out[0, :2]

    return (
        root_pos_out,
        local_body_pos.detach().cpu().numpy(),
        local_body_rot.detach().cpu().numpy(),
    )


def save_npz_atomic(path: pathlib.Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz",
        prefix=f"{path.stem}.",
        dir=path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = pathlib.Path(tmp_file.name)

    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def process_file(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    kinematics_model: KinematicsModel,
    *,
    height_adjust: bool,
    root_origin_offset: bool,
    no_anomaly_cut: bool,
    anomaly_margin: int,
    anomaly_threshold: float,
    min_segment_frames: int,
    dry_run: bool,
):
    with np.load(src_path, allow_pickle=True) as npz_data:
        motion_data = {key: npz_data[key] for key in npz_data.files}

    required_keys = ("root_pos", "root_rot", "dof_pos")
    missing = [key for key in required_keys if key not in motion_data]
    if missing:
        raise KeyError(f"missing required keys: {missing}")

    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])

    validate_motion_array("root_pos", root_pos, expected_dim=2, expected_len=root_pos.shape[0])
    validate_motion_array("root_rot", root_rot, expected_dim=2, expected_len=root_pos.shape[0])
    validate_motion_array("dof_pos", dof_pos, expected_dim=2, expected_len=root_pos.shape[0])

    if root_pos.shape[1] != 3:
        raise ValueError(f"root_pos shape mismatch: expected second dim 3, got {root_pos.shape}")
    if root_rot.shape[1] != 4:
        raise ValueError(f"root_rot shape mismatch: expected second dim 4, got {root_rot.shape}")
    if dof_pos.shape[1] != kinematics_model.num_dof:
        raise ValueError(
            f"dof_pos shape mismatch: got {dof_pos.shape[1]}, expected {kinematics_model.num_dof}"
        )

    max_delta = compute_max_deltas(dof_pos)
    stats = summarize_deltas(max_delta)
    threshold = float(anomaly_threshold)

    anomaly_frame_idx = np.zeros((0,), dtype=np.int64)
    anomaly_spans: list[tuple[int, int]] = []
    keep_spans: list[tuple[int, int]] = [(0, root_pos.shape[0])]
    if not no_anomaly_cut:
        anomaly_frame_idx, anomaly_spans = build_anomaly_spans(
            root_pos.shape[0],
            max_delta,
            threshold=threshold,
            anomaly_margin=anomaly_margin,
        )
        keep_spans = build_keep_spans(
            root_pos.shape[0],
            anomaly_spans,
            min_segment_frames=min_segment_frames,
        )
        if not keep_spans:
            raise ValueError(
                "no valid segments remain after anomaly trimming"
            )
    elif root_pos.shape[0] < min_segment_frames:
        raise ValueError("input motion is shorter than min_segment_frames")

    segment_infos = []
    frames_out_total = 0
    for seg_idx, (start, end) in enumerate(keep_spans):
        segment_data = slice_framewise_fields(motion_data, start, end)
        seg_root_pos = np.asarray(segment_data["root_pos"])
        seg_root_rot = np.asarray(segment_data["root_rot"])
        seg_dof_pos = np.asarray(segment_data["dof_pos"])

        rebuilt_root_pos, local_body_pos, local_body_rot = rebuild_motion_fields(
            kinematics_model,
            seg_root_pos,
            seg_root_rot,
            seg_dof_pos,
            height_adjust=height_adjust,
            root_origin_offset=root_origin_offset,
        )

        segment_data["root_pos"] = rebuilt_root_pos
        segment_data["local_body_pos"] = local_body_pos
        segment_data["local_body_rot"] = local_body_rot
        segment_data["body_names"] = np.asarray(kinematics_model.body_names)
        segment_data["joint_names"] = np.asarray(kinematics_model.joint_names)

        seg_dst_path = dst_path.parent / (
            f"{dst_path.stem}_seg{seg_idx:03d}_f{start:06d}_{end - 1:06d}{dst_path.suffix}"
        )
        if not dry_run:
            save_npz_atomic(seg_dst_path, segment_data)

        segment_infos.append(
            {
                "segment_index": seg_idx,
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "frames": int(end - start),
                "dst_file": str(seg_dst_path),
            }
        )
        frames_out_total += int(end - start)

    return {
        "src_file": str(src_path),
        "frames_in": int(root_pos.shape[0]),
        "frames_out_total": int(frames_out_total),
        "removed_frames": int(root_pos.shape[0] - frames_out_total),
        "body_count": int(kinematics_model.num_joint),
        "joint_count": int(kinematics_model.num_dof),
        "threshold": threshold,
        "threshold_mode": "fixed",
        "stats": stats,
        "anomaly_count": int(anomaly_frame_idx.shape[0]),
        "anomaly_spans": anomaly_spans,
        "keep_spans": keep_spans,
        "segments": segment_infos,
        "dry_run": dry_run,
    }


def main():
    args = parse_args()
    src_folder = args.src_folder.resolve()
    if not src_folder.exists():
        raise FileNotFoundError(f"src folder not found: {src_folder}")
    dst_folder = (
        args.dst_folder.resolve()
        if args.dst_folder is not None
        else src_folder.parent / f"{src_folder.name}_segmented"
    )

    device = choose_device(args.device)
    xml_file = str(ROBOT_XML_DICT[args.robot])
    kinematics_model = KinematicsModel(xml_file, device=device)

    npz_files = list_npz_files(src_folder)
    if not npz_files:
        raise FileNotFoundError(f"no .npz files found under: {src_folder}")

    print(f"Using robot xml: {xml_file}")
    print(f"Using device: {device}")
    print(f"Source folder: {src_folder}")
    print(f"Destination folder: {dst_folder}")
    print(f"Found {len(npz_files)} file(s) under {src_folder}")

    processed = []
    for file_path in npz_files:
        dst_path = dst_folder / file_path.relative_to(src_folder)
        info = process_file(
            file_path,
            dst_path,
            kinematics_model,
            height_adjust=not args.no_height_adjust,
            root_origin_offset=not args.no_root_origin_offset,
            no_anomaly_cut=args.no_anomaly_cut,
            anomaly_margin=args.anomaly_margin,
            anomaly_threshold=args.anomaly_threshold,
            min_segment_frames=args.min_segment_frames,
            dry_run=args.dry_run,
        )
        processed.append(info)
        mode = "validated" if args.dry_run else "updated"
        print(
            f"[{mode}] {file_path.name} "
            f"| frames {info['frames_in']} -> {info['frames_out_total']} "
            f"| removed={info['removed_frames']} "
            f"| threshold={info['threshold']:.4f} ({info['threshold_mode']}) "
            f"| anomalies={info['anomaly_count']} "
            f"| cut_spans={len(info['anomaly_spans'])} "
            f"| segments={len(info['segments'])}"
        )
        print(
            "  delta stats: "
            f"mean={info['stats']['mean']:.4f}, std={info['stats']['std']:.4f}, "
            f"median={info['stats']['median']:.4f}, q3={info['stats']['q3']:.4f}, "
            f"p99={info['stats']['p99']:.4f}, p99.5={info['stats']['p99_5']:.4f}, "
            f"p99.9={info['stats']['p99_9']:.4f}"
        )
        print(
            "  segment lengths: "
            + ", ".join(str(seg["frames"]) for seg in info["segments"][:12])
            + (" ..." if len(info["segments"]) > 12 else "")
        )

    print(f"Done. processed {len(processed)} file(s).")


if __name__ == "__main__":
    main()
