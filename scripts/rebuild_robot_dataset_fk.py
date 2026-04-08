#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

import mujoco
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
DEFAULT_GROUND_WINDOW = 200
DEFAULT_GROUND_PERCENTILE = 10.0
DEFAULT_HEIGHT_MODE = "trailing_p10"


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RebuildOptions:
    height_adjust_enabled: bool
    height_mode: str
    height_percentile: float
    root_origin_enabled: bool
    root_origin_xy: tuple[float, float]


@dataclass(frozen=True)
class FileSummary:
    src_path: pathlib.Path
    frames_in: int
    frames_out: int
    anomaly_count: int
    joint_reindexed: bool
    removed_ranges: list[FrameRange]
    kept_ranges: list[FrameRange]
    segment_paths: list[pathlib.Path]


@dataclass(frozen=True)
class MeshGeom:
    geom_id: int
    local_vertices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut anomalous motion spans with a fixed dof_pos threshold, split the "
            "remaining motion into valid segments, then rebuild local FK fields."
        )
    )
    parser.add_argument(
        "--src-folder",
        type=pathlib.Path,
        default=DEFAULT_SRC_FOLDER,
        help="Folder containing input .npz files. Processed recursively.",
    )
    parser.add_argument(
        "--dst-folder",
        type=pathlib.Path,
        default=None,
        help="Output folder. Defaults to <src>_segmented next to the source folder.",
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
        help='Torch device, for example "auto", "cpu", or "cuda:0".',
    )
    parser.add_argument(
        "--disable-anomaly-cut",
        action="store_true",
        help="Keep the full input motion instead of cutting anomalous spans.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.6,
        help="Cut when max(abs(dof_pos[t] - dof_pos[t-1])) exceeds this threshold.",
    )
    parser.add_argument(
        "--anomaly-margin",
        type=int,
        default=25,
        help="Remove this many frames before and after each detected anomaly frame.",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=30,
        help="Drop kept segments shorter than this many frames.",
    )
    parser.add_argument(
        "--disable-height-adjust",
        action="store_true",
        help=(
            "Do not shift root_pos by a mesh-ground estimate."
        ),
    )
    parser.add_argument(
        "--height-mode",
        type=str,
        default=DEFAULT_HEIGHT_MODE,
        choices=("trailing_p10", "current_frame_min"),
        help=(
            "Height adjustment mode. "
            "'trailing_p10' uses the past 200-frame percentile of the global mesh minimum; "
            "'current_frame_min' aligns each frame's own global mesh minimum to z=0."
        ),
    )
    parser.add_argument(
        "--height-percentile",
        type=float,
        default=DEFAULT_GROUND_PERCENTILE,
        help=(
            "Percentile used by trailing_p10 mode over the past 200 frames. "
            "Set 10 for p10, 50 for p50, and so on."
        ),
    )
    parser.add_argument(
        "--disable-root-origin-offset",
        action="store_true",
        help="Do not translate root_pos so the first frame XY lands at the target XY.",
    )
    parser.add_argument(
        "--root-origin-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(0.0, 0.0),
        help="Target XY for the first root frame when root origin offset is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline without writing any files.",
    )
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> RebuildOptions:
    return RebuildOptions(
        height_adjust_enabled=not args.disable_height_adjust,
        height_mode=args.height_mode,
        height_percentile=float(args.height_percentile),
        root_origin_enabled=not args.disable_root_origin_offset,
        root_origin_xy=(float(args.root_origin_xy[0]), float(args.root_origin_xy[1])),
    )


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def list_npz_files(src_folder: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in src_folder.rglob("*.npz") if path.is_file())


def load_motion_payload(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz_data:
        return {key: npz_data[key] for key in npz_data.files}


def validate_motion_payload(
    motion_data: dict[str, Any],
    *,
    expected_dof: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_keys = ("root_pos", "root_rot", "dof_pos")
    missing = [key for key in required_keys if key not in motion_data]
    if missing:
        raise KeyError(f"missing required keys: {missing}")

    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T, 3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_rot must have shape (T, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[0] != root_pos.shape[0]:
        raise ValueError(
            f"dof_pos must have shape (T, num_dof), got {dof_pos.shape}"
        )
    if dof_pos.shape[1] != expected_dof:
        raise ValueError(f"dof_pos second dim must be {expected_dof}, got {dof_pos.shape[1]}")

    return root_pos, root_rot, dof_pos


def normalize_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def maybe_reindex_dof_pos(
    motion_data: dict[str, Any],
    dof_pos: np.ndarray,
    *,
    expected_joint_names: list[str],
) -> tuple[np.ndarray, bool]:
    input_joint_names = motion_data.get("joint_names")
    if input_joint_names is None:
        return dof_pos, False

    names_in = [normalize_name(item) for item in np.asarray(input_joint_names).reshape(-1)]
    if len(names_in) != len(expected_joint_names):
        raise ValueError(
            f"joint_names length mismatch: got {len(names_in)}, expected {len(expected_joint_names)}"
        )
    if names_in == expected_joint_names:
        return dof_pos, False

    name_to_index: dict[str, int] = {}
    for index, name in enumerate(names_in):
        if name in name_to_index:
            raise ValueError(f"joint_names contains duplicates: {name}")
        name_to_index[name] = index

    missing = [name for name in expected_joint_names if name not in name_to_index]
    extra = [name for name in names_in if name not in set(expected_joint_names)]
    if missing or extra:
        raise ValueError(
            f"joint_names do not match robot standard order. missing={missing}, extra={extra}"
        )

    reorder_index = [name_to_index[name] for name in expected_joint_names]
    return dof_pos[:, reorder_index], True


def compute_frame_jump_scores(dof_pos: np.ndarray) -> np.ndarray:
    if dof_pos.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float64)
    delta = np.diff(np.asarray(dof_pos, dtype=np.float64), axis=0)
    return np.max(np.abs(delta), axis=1)


def load_mesh_geoms(xml_file: str) -> tuple[mujoco.MjModel, list[MeshGeom]]:
    model = mujoco.MjModel.from_xml_path(xml_file)
    mesh_geoms = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = np.array(model.mesh_vert[start : start + count], dtype=np.float32)
        mesh_geoms.append(MeshGeom(geom_id=geom_id, local_vertices=verts))

    if not mesh_geoms:
        raise ValueError(f"no mesh geoms found in robot xml: {xml_file}")
    return model, mesh_geoms


def compute_trailing_percentile(
    values: np.ndarray,
    *,
    window: int,
    percentile: float,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)

    out = np.empty_like(values, dtype=np.float32)
    prefix_count = min(window - 1, len(values))
    for idx in range(prefix_count):
        out[idx] = np.percentile(values[: idx + 1], percentile)

    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
        out[window - 1 :] = np.percentile(windows, percentile, axis=-1)

    return out


def compute_mesh_ground_minima(
    mj_model: mujoco.MjModel,
    mesh_geoms: list[MeshGeom],
    *,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    mj_data = mujoco.MjData(mj_model)
    root_rot_wxyz = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=1)
    minima = np.empty((root_pos.shape[0],), dtype=np.float32)

    for frame_idx in range(root_pos.shape[0]):
        mj_data.qpos[:3] = root_pos[frame_idx]
        mj_data.qpos[3:7] = root_rot_wxyz[frame_idx]
        mj_data.qpos[7:] = dof_pos[frame_idx]
        mujoco.mj_forward(mj_model, mj_data)

        frame_min = np.inf
        for mesh_geom in mesh_geoms:
            geom_rot = mj_data.geom_xmat[mesh_geom.geom_id].reshape(3, 3)
            geom_pos = mj_data.geom_xpos[mesh_geom.geom_id]
            geom_min = np.min(mesh_geom.local_vertices @ geom_rot[2] + geom_pos[2])
            frame_min = min(frame_min, float(geom_min))
        minima[frame_idx] = frame_min

    return minima


def compute_ground_baseline(
    mesh_minima: np.ndarray,
    *,
    height_mode: str,
    height_percentile: float,
) -> np.ndarray:
    if height_mode == "current_frame_min":
        return np.asarray(mesh_minima, dtype=np.float32)
    if height_mode == "trailing_p10":
        return compute_trailing_percentile(
            mesh_minima,
            window=DEFAULT_GROUND_WINDOW,
            percentile=height_percentile,
        )
    raise ValueError(f"unsupported height mode: {height_mode}")


def merge_ranges(ranges: list[FrameRange]) -> list[FrameRange]:
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda item: (item.start, item.end))
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = FrameRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return merged


def detect_anomaly_ranges(
    *,
    num_frames: int,
    jump_scores: np.ndarray,
    threshold: float,
    margin: int,
) -> tuple[np.ndarray, list[FrameRange]]:
    anomaly_frames = np.where(jump_scores > threshold)[0] + 1
    ranges = []
    for frame_idx in anomaly_frames:
        start = max(0, int(frame_idx) - margin)
        end = min(num_frames, int(frame_idx) + margin + 1)
        ranges.append(FrameRange(start, end))
    return anomaly_frames, merge_ranges(ranges)


def compute_kept_ranges(
    *,
    num_frames: int,
    removed_ranges: list[FrameRange],
    min_segment_frames: int,
) -> list[FrameRange]:
    if not removed_ranges:
        return [FrameRange(0, num_frames)] if num_frames > 0 else []

    kept_ranges = []
    cursor = 0
    for removed in removed_ranges:
        if removed.start > cursor:
            kept = FrameRange(cursor, removed.start)
            if kept.frames >= min_segment_frames:
                kept_ranges.append(kept)
        cursor = max(cursor, removed.end)

    if cursor < num_frames:
        kept = FrameRange(cursor, num_frames)
        if kept.frames >= min_segment_frames:
            kept_ranges.append(kept)

    return kept_ranges


def slice_framewise_payload(
    motion_data: dict[str, Any],
    frame_range: FrameRange,
) -> dict[str, Any]:
    num_frames = int(np.asarray(motion_data["root_pos"]).shape[0])
    sliced = {}
    for key, value in motion_data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == num_frames:
            sliced[key] = value[frame_range.start:frame_range.end]
        else:
            sliced[key] = value
    return sliced


def rebuild_fk_fields(
    kinematics_model: KinematicsModel,
    mj_model: mujoco.MjModel,
    mesh_geoms: list[MeshGeom],
    *,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    options: RebuildOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = int(root_pos.shape[0])
    device = kinematics_model._device

    root_pos_out = np.array(root_pos, copy=True)
    dof_tensor = torch.from_numpy(np.asarray(dof_pos)).to(device=device, dtype=torch.float32)

    fk_root_pos = torch.zeros((num_frames, 3), device=device, dtype=torch.float32)
    fk_root_rot = torch.zeros((num_frames, 4), device=device, dtype=torch.float32)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, local_body_rot = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        dof_tensor,
    )

    if options.height_adjust_enabled:
        mesh_minima = compute_mesh_ground_minima(
            mj_model,
            mesh_geoms,
            root_pos=root_pos_out,
            root_rot_xyzw=np.asarray(root_rot, dtype=np.float32),
            dof_pos=np.asarray(dof_pos, dtype=np.float32),
        )
        ground_baseline = compute_ground_baseline(
            mesh_minima,
            height_mode=options.height_mode,
            height_percentile=options.height_percentile,
        )
        root_pos_out[:, 2] -= ground_baseline

    if options.root_origin_enabled and num_frames > 0:
        target_xy = np.asarray(options.root_origin_xy, dtype=root_pos_out.dtype)
        root_pos_out[:, :2] += target_xy - root_pos_out[0, :2]

    return (
        root_pos_out,
        local_body_pos.detach().cpu().numpy(),
        local_body_rot.detach().cpu().numpy(),
    )


def build_output_path(dst_path: pathlib.Path, frame_range: FrameRange, segment_index: int) -> pathlib.Path:
    return dst_path.parent / (
        f"{dst_path.stem}_seg{segment_index:03d}_"
        f"f{frame_range.start:06d}_{frame_range.end - 1:06d}{dst_path.suffix}"
    )


def save_npz_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
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


def process_motion_file(
    *,
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    kinematics_model: KinematicsModel,
    mj_model: mujoco.MjModel,
    mesh_geoms: list[MeshGeom],
    options: RebuildOptions,
    anomaly_cut_enabled: bool,
    anomaly_threshold: float,
    anomaly_margin: int,
    min_segment_frames: int,
    dry_run: bool,
) -> FileSummary:
    motion_data = load_motion_payload(src_path)
    root_pos, root_rot, dof_pos = validate_motion_payload(
        motion_data,
        expected_dof=kinematics_model.num_dof,
    )
    dof_pos, joint_reindexed = maybe_reindex_dof_pos(
        motion_data,
        dof_pos,
        expected_joint_names=list(kinematics_model.joint_names),
    )
    motion_data = dict(motion_data)
    motion_data["dof_pos"] = dof_pos
    motion_data["joint_names"] = np.asarray(kinematics_model.joint_names)

    removed_ranges: list[FrameRange] = []
    anomaly_frames = np.zeros((0,), dtype=np.int64)
    if anomaly_cut_enabled:
        jump_scores = compute_frame_jump_scores(dof_pos)
        anomaly_frames, removed_ranges = detect_anomaly_ranges(
            num_frames=root_pos.shape[0],
            jump_scores=jump_scores,
            threshold=anomaly_threshold,
            margin=anomaly_margin,
        )
        kept_ranges = compute_kept_ranges(
            num_frames=root_pos.shape[0],
            removed_ranges=removed_ranges,
            min_segment_frames=min_segment_frames,
        )
    else:
        kept_ranges = [FrameRange(0, root_pos.shape[0])] if root_pos.shape[0] > 0 else []

    if not kept_ranges:
        raise ValueError("no valid segments remain after filtering")

    segment_paths = []
    frames_out = 0
    for segment_index, frame_range in enumerate(kept_ranges):
        segment_data = slice_framewise_payload(motion_data, frame_range)
        rebuilt_root_pos, local_body_pos, local_body_rot = rebuild_fk_fields(
            kinematics_model,
            mj_model,
            mesh_geoms,
            root_pos=np.asarray(segment_data["root_pos"]),
            root_rot=np.asarray(segment_data["root_rot"]),
            dof_pos=np.asarray(segment_data["dof_pos"]),
            options=options,
        )

        segment_data["root_pos"] = rebuilt_root_pos
        segment_data["local_body_pos"] = local_body_pos
        segment_data["local_body_rot"] = local_body_rot
        segment_data["body_names"] = np.asarray(kinematics_model.body_names)
        segment_data["joint_names"] = np.asarray(kinematics_model.joint_names)

        if not anomaly_cut_enabled:
            output_path = dst_path
        else:
            output_path = build_output_path(dst_path, frame_range, segment_index)
        if not dry_run:
            save_npz_atomic(output_path, segment_data)

        segment_paths.append(output_path)
        frames_out += frame_range.frames

    return FileSummary(
        src_path=src_path,
        frames_in=int(root_pos.shape[0]),
        frames_out=int(frames_out),
        anomaly_count=int(anomaly_frames.shape[0]),
        joint_reindexed=joint_reindexed,
        removed_ranges=removed_ranges,
        kept_ranges=kept_ranges,
        segment_paths=segment_paths,
    )


def format_ranges(ranges: list[FrameRange], limit: int = 6) -> str:
    if not ranges:
        return "-"
    text = ", ".join(f"[{item.start},{item.end})" for item in ranges[:limit])
    if len(ranges) > limit:
        text += ", ..."
    return text


def main() -> None:
    args = parse_args()
    src_folder = args.src_folder.resolve()
    if not src_folder.exists():
        raise FileNotFoundError(f"source folder not found: {src_folder}")

    dst_folder = (
        args.dst_folder.resolve()
        if args.dst_folder is not None
        else src_folder.parent / f"{src_folder.name}_segmented"
    )
    device = choose_device(args.device)
    xml_file = str(ROBOT_XML_DICT[args.robot])
    options = build_options(args)

    npz_files = list_npz_files(src_folder)
    if not npz_files:
        raise FileNotFoundError(f"no .npz files found under: {src_folder}")

    kinematics_model = KinematicsModel(xml_file, device=device)
    mj_model, mesh_geoms = load_mesh_geoms(xml_file)

    print(f"Using robot xml: {xml_file}")
    print(f"Using device: {device}")
    print(f"Source folder: {src_folder}")
    print(f"Destination folder: {dst_folder}")
    print(f"Found {len(npz_files)} file(s)")
    height_mode_label = options.height_mode
    if options.height_mode == "trailing_p10":
        height_mode_label += f",{DEFAULT_GROUND_WINDOW}f,p{options.height_percentile:.1f}"

    print(
        "Options: "
        f"anomaly_cut={'on' if not args.disable_anomaly_cut else 'off'}, "
        f"height_adjust={'on' if options.height_adjust_enabled else 'off'}"
        f"->mesh-ground({height_mode_label})->z=0.0000, "
        f"root_origin={'on' if options.root_origin_enabled else 'off'}"
        f"->xy=({options.root_origin_xy[0]:.4f}, {options.root_origin_xy[1]:.4f}), "
        f"min_segment_frames={args.min_segment_frames}"
    )

    processed_count = 0
    for src_path in npz_files:
        dst_path = dst_folder / src_path.relative_to(src_folder)
        summary = process_motion_file(
            src_path=src_path,
            dst_path=dst_path,
            kinematics_model=kinematics_model,
            mj_model=mj_model,
            mesh_geoms=mesh_geoms,
            options=options,
            anomaly_cut_enabled=not args.disable_anomaly_cut,
            anomaly_threshold=args.anomaly_threshold,
            anomaly_margin=args.anomaly_margin,
            min_segment_frames=args.min_segment_frames,
            dry_run=args.dry_run,
        )
        processed_count += 1

        mode = "validated" if args.dry_run else "written"
        print(
            f"[{mode}] {src_path.name} | frames {summary.frames_in} -> {summary.frames_out} "
            f"| segments={len(summary.kept_ranges)} | anomalies={summary.anomaly_count} "
            f"| removed={summary.frames_in - summary.frames_out}"
            f"{' | reindexed-joints' if summary.joint_reindexed else ''}"
        )
        if not args.disable_anomaly_cut:
            print(f"  removed ranges: {format_ranges(summary.removed_ranges)}")
        print(f"  kept ranges: {format_ranges(summary.kept_ranges)}")

    print(f"Done. processed {processed_count} file(s).")


if __name__ == "__main__":
    main()
