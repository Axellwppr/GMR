import argparse
import copy
import json
import pathlib
import time
from typing import Any

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from utils.data_processor import load_robot_init
from utils.fk_solver import MuJoCoFK

XROBOT_FROM_SMPLX_DELTA_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

BVH_XSENS_BONE_NAME_MAP = {
    "Hips": "Hips",
    "Spine2": "Chest4",
    "LeftUpLeg": "LeftHip",
    "RightUpLeg": "RightHip",
    "LeftLeg": "LeftKnee",
    "RightLeg": "RightKnee",
    "LeftFootMod": "LeftFootMod",
    "RightFootMod": "RightFootMod",
    "LeftArm": "LeftShoulder",
    "RightArm": "RightShoulder",
    "LeftForeArm": "LeftElbow",
    "RightForeArm": "RightElbow",
    "LeftHand": "LeftWrist",
    "RightHand": "RightWrist",
}

BVH_LAFAN_TO_XSENS_DELTA_WXYZ = {
    "Hips": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    "Spine2": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    "LeftUpLeg": np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64),
    "RightUpLeg": np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64),
    "LeftLeg": np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64),
    "RightLeg": np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64),
    "LeftFootMod": np.array([0.70710678, -0.70710678, 0.0, 0.0], dtype=np.float64),
    "RightFootMod": np.array([0.70710678, -0.70710678, 0.0, 0.0], dtype=np.float64),
    "LeftArm": np.array([0.70710678, 0.0, 0.0, -0.70710678], dtype=np.float64),
    "LeftForeArm": np.array([0.70710678, 0.0, 0.0, -0.70710678], dtype=np.float64),
    "LeftHand": np.array([0.70710678, 0.0, 0.0, -0.70710678], dtype=np.float64),
    "RightArm": np.array([0.0, 0.70710678, -0.70710678, 0.0], dtype=np.float64),
    "RightForeArm": np.array([0.0, 0.70710678, -0.70710678, 0.0], dtype=np.float64),
    "RightHand": np.array([0.0, 0.70710678, -0.70710678, 0.0], dtype=np.float64),
}


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion norm is too small.")
    return q / n


def _load_human_tpose_frame(src_human: str, smplx_file: str | None, bvh_file: str | None, smplx_model_dir: str):
    if src_human == "smplx":
        if smplx_file is None:
            raise ValueError("--smplx_file is required when --src_human smplx")

        smplx_data, body_model, _, actual_human_height = load_smplx_file(smplx_file, smplx_model_dir)
        frames, _ = get_smplx_data_offline_fast(smplx_data, body_model, tgt_fps=30)
        if len(frames) == 0:
            raise RuntimeError("No frame found in SMPLX file.")
        return frames[0], float(actual_human_height)

    if src_human == "bvh":
        if bvh_file is None:
            raise ValueError("--bvh_file is required when --src_human bvh")

        frames, actual_human_height = load_lafan1_file(bvh_file)
        if len(frames) == 0:
            raise RuntimeError("No frame found in BVH file.")
        return frames[0], float(actual_human_height)

    raise ValueError(f"Unsupported src_human: {src_human}")


def _to_body_name_to_index(body_names: list[Any]) -> dict[str, int]:
    out = {}
    for idx, name in enumerate(body_names):
        if not isinstance(name, str):
            name = name.decode("utf-8")
        out[name] = idx
    return out


def _extract_weights(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    t1_pos = float(entry.get("table1_pos", entry.get("pos_weight", 0.0)))
    t1_rot = float(entry.get("table1_rot", entry.get("rot_weight", 0.0)))
    t2_pos = float(entry.get("table2_pos", t1_pos))
    t2_rot = float(entry.get("table2_rot", t1_rot))
    return t1_pos, t1_rot, t2_pos, t2_rot


def _extract_pos_offsets(entry: dict[str, Any]) -> tuple[list[float], list[float]]:
    default_offset = [0.0, 0.0, 0.0]
    shared_offset = entry.get("pos_offset", default_offset)
    table1_offset = entry.get("table1_pos_offset", shared_offset)
    table2_offset = entry.get("table2_pos_offset", shared_offset)

    def _to_xyz_list(value: Any) -> list[float]:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"Position offset must have exactly 3 values, got: {value}")
        return arr.tolist()

    return _to_xyz_list(table1_offset), _to_xyz_list(table2_offset)


def _build_default_human_scale_table(mappings: list[dict[str, Any]], human_root_name: str) -> dict[str, float]:
    human_bones = {human_root_name}
    for entry in mappings:
        human_bones.add(entry["human_bone"])
    return {name: 1.0 for name in sorted(human_bones)}


def _maybe_get_quat(entry: dict[str, Any], key_candidates: list[str]) -> np.ndarray | None:
    for k in key_candidates:
        if k in entry and entry[k] is not None:
            return _normalize_quat_wxyz(np.asarray(entry[k], dtype=np.float64).reshape(4))
    return None


def _smplx_bone_to_xrobot_bone(name: str) -> str:
    parts = str(name).split("_")
    return "_".join(part[:1].upper() + part[1:] for part in parts if part != "")


def _left_multiply_quat_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    q = (
        R.from_quat(_normalize_quat_wxyz(lhs), scalar_first=True)
        * R.from_quat(_normalize_quat_wxyz(rhs), scalar_first=True)
    ).as_quat(scalar_first=True)
    return _normalize_quat_wxyz(q)


def _inverse_quat_wxyz(q: np.ndarray) -> np.ndarray:
    return _normalize_quat_wxyz(
        R.from_quat(_normalize_quat_wxyz(q), scalar_first=True).inv().as_quat(scalar_first=True)
    )


def _derive_xrobot_output_path(output_path: pathlib.Path) -> pathlib.Path:
    name = output_path.name
    if "smplx_to_" in name:
        return output_path.with_name(name.replace("smplx_to_", "xrobot_to_", 1))
    if name.startswith("smplx_"):
        return output_path.with_name(name.replace("smplx_", "xrobot_", 1))
    return output_path.with_name(f"{output_path.stem}_xrobot{output_path.suffix}")


def _derive_bvh_xsens_output_path(output_path: pathlib.Path) -> pathlib.Path:
    name = output_path.name
    if "bvh_lafan1_to_" in name:
        return output_path.with_name(name.replace("bvh_lafan1_to_", "bvh_xsens_to_", 1))
    if "bvh_to_" in name:
        return output_path.with_name(name.replace("bvh_to_", "bvh_xsens_to_", 1))
    if name.startswith("bvh_lafan1_"):
        return output_path.with_name(name.replace("bvh_lafan1_", "bvh_xsens_", 1))
    if name.startswith("bvh_"):
        return output_path.with_name(name.replace("bvh_", "bvh_xsens_", 1))
    return output_path.with_name(f"{output_path.stem}_xsens{output_path.suffix}")


def _build_xrobot_cfg_from_smplx_cfg(output_cfg: dict[str, Any]) -> dict[str, Any]:
    xrobot_cfg = copy.deepcopy(output_cfg)
    xrobot_cfg["human_root_name"] = _smplx_bone_to_xrobot_bone(output_cfg["human_root_name"])
    xrobot_cfg["human_scale_table"] = {
        _smplx_bone_to_xrobot_bone(human_bone): scale
        for human_bone, scale in output_cfg["human_scale_table"].items()
    }

    for table_key in ("ik_match_table1", "ik_match_table2"):
        updated_table = {}
        for robot_link, entry in xrobot_cfg[table_key].items():
            human_bone, pos_w, rot_w, pos_offset, quat_offset = entry
            quat_offset_xrobot = _left_multiply_quat_wxyz(
                XROBOT_FROM_SMPLX_DELTA_WXYZ,
                np.asarray(quat_offset, dtype=np.float64),
            ).tolist()
            updated_table[robot_link] = [
                _smplx_bone_to_xrobot_bone(human_bone),
                pos_w,
                rot_w,
                pos_offset,
                quat_offset_xrobot,
            ]
        xrobot_cfg[table_key] = updated_table

    return xrobot_cfg


def _build_xsens_cfg_from_bvh_cfg(
    output_cfg: dict[str, Any],
    human_scale_table_xsens_lafan_keys: dict[str, float] | None,
) -> dict[str, Any]:
    xsens_cfg = copy.deepcopy(output_cfg)
    xsens_root_name = BVH_XSENS_BONE_NAME_MAP.get(output_cfg["human_root_name"], output_cfg["human_root_name"])
    xsens_cfg["human_root_name"] = xsens_root_name

    scale_src = (
        {str(k): float(v) for k, v in human_scale_table_xsens_lafan_keys.items()}
        if human_scale_table_xsens_lafan_keys is not None
        else {str(k): float(v) for k, v in output_cfg["human_scale_table"].items()}
    )
    xsens_cfg["human_scale_table"] = {
        BVH_XSENS_BONE_NAME_MAP.get(lafan_bone, lafan_bone): scale
        for lafan_bone, scale in scale_src.items()
    }

    for table_key in ("ik_match_table1", "ik_match_table2"):
        updated_table = {}
        for robot_link, entry in xsens_cfg[table_key].items():
            lafan_bone, pos_w, rot_w, pos_offset, quat_offset = entry
            xsens_bone = BVH_XSENS_BONE_NAME_MAP.get(lafan_bone, lafan_bone)
            delta = BVH_LAFAN_TO_XSENS_DELTA_WXYZ.get(lafan_bone, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
            quat_offset_xsens = _left_multiply_quat_wxyz(
                _inverse_quat_wxyz(delta),
                np.asarray(quat_offset, dtype=np.float64),
            ).tolist()
            updated_table[robot_link] = [
                xsens_bone,
                pos_w,
                rot_w,
                pos_offset,
                quat_offset_xsens,
            ]
        xsens_cfg[table_key] = updated_table

    return xsens_cfg


def _draw_axes(
    viewer,
    pos: np.ndarray,
    rot_mat: np.ndarray,
    axis_scale: float,
    alpha: float,
    label: str | None = None,
):
    base_colors = np.array(
        [
            [1.0, 0.0, 0.0, alpha],  # X
            [0.0, 1.0, 0.0, alpha],  # Y
            [0.0, 0.0, 1.0, alpha],  # Z
        ],
        dtype=np.float64,
    )
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    mat = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)
    for i in range(3):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos,
            mat=mat.flatten(),
            rgba=base_colors[i],
        )
        if label is not None:
            geom.label = label
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.004,
            from_=pos,
            to=pos + axis_scale * mat[:, i],
        )
        viewer.user_scn.ngeom += 1


def _to_numpy_human_frame(human_frame: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body_name, value in human_frame.items():
        pos = np.asarray(value[0], dtype=np.float64).reshape(3)
        quat = _normalize_quat_wxyz(np.asarray(value[1], dtype=np.float64).reshape(4))
        out[body_name] = (pos, quat)
    return out


def _scale_human_frame(
    human_frame: dict[str, tuple[np.ndarray, np.ndarray]],
    human_root_name: str,
    human_scale_table: dict[str, float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if human_root_name not in human_frame:
        raise KeyError(f"human_root_name '{human_root_name}' not found in human frame.")

    root_pos, root_quat = human_frame[human_root_name]
    root_scale = float(human_scale_table.get(human_root_name, 1.0))
    scaled_root_pos = root_scale * root_pos

    scaled_human: dict[str, tuple[np.ndarray, np.ndarray]] = {
        human_root_name: (scaled_root_pos, root_quat)
    }
    for body_name, (pos, quat) in human_frame.items():
        if body_name == human_root_name:
            continue
        if body_name not in human_scale_table:
            continue
        scale = float(human_scale_table[body_name])
        scaled_pos = (pos - root_pos) * scale + scaled_root_pos
        scaled_human[body_name] = (scaled_pos, quat)
    return scaled_human


def _build_corrected_human_frame_for_preview(
    scaled_human_frame: dict[str, tuple[np.ndarray, np.ndarray]],
    ik_match_table1: dict[str, list[Any]],
    ground_height: float,
) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    ground = np.array([0.0, 0.0, float(ground_height)], dtype=np.float64)
    corrected: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for robot_link, entry in ik_match_table1.items():
        human_bone, _, _, pos_offset, quat_offset = entry
        if human_bone not in scaled_human_frame:
            continue

        pos, quat = scaled_human_frame[human_bone]
        updated_quat = (
            R.from_quat(quat, scalar_first=True) * R.from_quat(np.asarray(quat_offset, dtype=np.float64), scalar_first=True)
        ).as_quat(scalar_first=True)
        updated_quat = _normalize_quat_wxyz(updated_quat)

        local_offset = np.asarray(pos_offset, dtype=np.float64) - ground
        global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
        corrected_pos = pos + global_pos_offset
        corrected[robot_link] = (human_bone, corrected_pos, updated_quat)
    return corrected


def _preview_tpose_axes(
    robot_xml: pathlib.Path,
    qpos_fk: np.ndarray,
    human_frame: dict[str, Any],
    human_scale_table: dict[str, float],
    ground_height: float,
    human_root_name: str,
    robot_root_name: str,
    ik_match_table1: dict[str, list[Any]],
    preview_seconds: float,
    axis_scale: float,
    show_human_raw_axes: bool,
    show_corrected_axes: bool,
):
    model = mj.MjModel.from_xml_path(str(robot_xml))
    data = mj.MjData(model)
    data.qpos[:] = np.asarray(qpos_fk, dtype=np.float64)
    mj.mj_forward(model, data)

    body_id_by_name = {}
    for i in range(model.nbody):
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if bname is not None:
            body_id_by_name[bname] = i

    root_robot_idx = body_id_by_name.get(robot_root_name, None)
    if root_robot_idx is None:
        raise KeyError(f"robot_root_name '{robot_root_name}' not found in MuJoCo model bodies.")
    root_robot_pos = np.asarray(data.xpos[root_robot_idx], dtype=np.float64)

    if human_root_name not in human_frame:
        raise KeyError(f"human_root_name '{human_root_name}' not found in human frame.")
    human_frame_np = _to_numpy_human_frame(human_frame)
    scaled_human_frame = _scale_human_frame(human_frame_np, human_root_name, human_scale_table)
    corrected_human_frame = _build_corrected_human_frame_for_preview(
        scaled_human_frame=scaled_human_frame,
        ik_match_table1=ik_match_table1,
        ground_height=ground_height,
    )

    # Preview alignment rule:
    # Anchor human root to robot root in world space for direct visual comparison.
    if human_root_name not in scaled_human_frame:
        raise KeyError(f"Cannot find human root '{human_root_name}' in scaled human frame.")
    scaled_root_pos, _ = scaled_human_frame[human_root_name]
    human_align_offset = root_robot_pos - scaled_root_pos

    # Keep corrected axes on the same root anchor (do not let corrected root drift visually).
    corrected_root_pos = None
    if robot_root_name in corrected_human_frame:
        _, corrected_root_pos, _ = corrected_human_frame[robot_root_name]
    else:
        for _, (hb, cp, _) in corrected_human_frame.items():
            if hb == human_root_name:
                corrected_root_pos = cp
                break
    corrected_reanchor_offset = np.zeros(3, dtype=np.float64)
    if corrected_root_pos is not None:
        corrected_reanchor_offset = scaled_root_pos - corrected_root_pos

    with mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Make robot semi-transparent so mapped axes are visible.
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        start = time.time()
        while viewer.is_running() and (time.time() - start) < preview_seconds:
            data.qpos[:] = np.asarray(qpos_fk, dtype=np.float64)
            mj.mj_forward(model, data)
            viewer.user_scn.ngeom = 0

            for robot_link, entry in ik_match_table1.items():
                human_bone, _, _, _, _ = entry

                body_id = body_id_by_name.get(robot_link, None)
                if body_id is None:
                    continue
                if human_bone not in human_frame:
                    continue

                robot_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
                robot_mat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                _draw_axes(viewer, robot_pos, robot_mat, axis_scale=axis_scale, alpha=1.0, label=robot_link)

                if show_human_raw_axes:
                    if human_bone in scaled_human_frame:
                        h_pos, h_quat = scaled_human_frame[human_bone]
                        h_mat = R.from_quat(h_quat, scalar_first=True).as_matrix()
                        _draw_axes(viewer, h_pos + human_align_offset, h_mat, axis_scale=axis_scale * 0.8, alpha=0.45)

                if show_corrected_axes:
                    if robot_link in corrected_human_frame:
                        _, corrected_pos, corrected_quat = corrected_human_frame[robot_link]
                        corrected_mat = R.from_quat(corrected_quat, scalar_first=True).as_matrix()
                        _draw_axes(
                            viewer,
                            corrected_pos + corrected_reanchor_offset + human_align_offset,
                            corrected_mat,
                            axis_scale=axis_scale * 0.9,
                            alpha=0.65,
                        )

            viewer.sync()
            time.sleep(1.0 / 60.0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build IK config directly from a simple mapping config + robot pose init + T-pose human skeleton. "
            "Quaternion offsets are computed automatically; optional position offsets can be provided in mappings."
        )
    )
    parser.add_argument("--robot_name", type=str, default="custom_robot", help="Only for logging/output metadata.")
    parser.add_argument("--robot_xml", type=str, required=True, help="Path to robot MJCF/XML file.")
    parser.add_argument("--src_human", type=str, choices=["smplx", "bvh"], required=True)

    parser.add_argument("--smplx_file", type=str, default=None)
    parser.add_argument("--bvh_file", type=str, default=None)
    parser.add_argument(
        "--smplx_model_dir",
        type=str,
        default=str(pathlib.Path(__file__).parent.parent / "assets" / "body_models"),
        help="Path to SMPLX body model directory (the folder containing 'smplx').",
    )

    parser.add_argument("--pose_init", type=str, required=True, help="Path to robot T-pose init JSON")
    parser.add_argument("--mapping_config", type=str, required=True, help="Simple mapping JSON path")
    parser.add_argument("--output", type=str, required=True, help="Output IK config JSON path")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        default=False,
        help="Skip missing robot links / human bones instead of failing.",
    )
    parser.add_argument("--preview", action="store_true", default=False, help="Preview T-pose axes after generation.")
    parser.add_argument("--preview_seconds", type=float, default=60.0, help="Preview duration in seconds.")
    parser.add_argument("--preview_axis_scale", type=float, default=0.08, help="Arrow length for axis visualization.")
    parser.add_argument(
        "--preview_show_human_raw_axes",
        action="store_true",
        default=False,
        help="Also draw aligned raw human joint axes.",
    )
    parser.add_argument(
        "--preview_show_corrected_axes",
        action="store_true",
        default=False,
        help="Also draw human axes after quat offset correction.",
    )

    args = parser.parse_args()

    robot_xml = pathlib.Path(args.robot_xml)
    if not robot_xml.exists():
        raise FileNotFoundError(f"robot_xml not found: {robot_xml}")

    with open(args.mapping_config, "r", encoding="utf-8") as f:
        simple_cfg = json.load(f)

    mappings = simple_cfg.get("mappings", None)
    if not isinstance(mappings, list) or len(mappings) == 0:
        raise ValueError("mapping_config must contain non-empty 'mappings' list.")

    human_root_name = simple_cfg.get("human_root_name", None)
    robot_root_name = simple_cfg.get("robot_root_name", None)
    if not human_root_name or not robot_root_name:
        raise ValueError("mapping_config must contain 'human_root_name' and 'robot_root_name'.")
    # Optional root-specific quat controls for convention mismatch handling.
    root_quat_override = _maybe_get_quat(
        simple_cfg,
        ["root_quat_offset_override_wxyz", "root_quat_offset_override"],
    )
    root_quat_extra = _maybe_get_quat(
        simple_cfg,
        ["root_quat_extra_wxyz", "root_quat_extra"],
    )

    human_frame, actual_human_height = _load_human_tpose_frame(
        src_human=args.src_human,
        smplx_file=args.smplx_file,
        bvh_file=args.bvh_file,
        smplx_model_dir=args.smplx_model_dir,
    )

    root_pos, root_rot, joints_dict, _ = load_robot_init(args.pose_init)
    if not isinstance(joints_dict, dict) or len(joints_dict) == 0:
        raise ValueError(f"No joints found in pose init: {args.pose_init}")

    fk_solver = MuJoCoFK(str(robot_xml))
    joint_order = fk_solver.joint_order

    dof_vec = np.zeros(len(joint_order), dtype=np.float64)
    assigned = 0
    for i, joint_name in enumerate(joint_order):
        if joint_name in joints_dict:
            dof_vec[i] = float(joints_dict[joint_name])
            assigned += 1

    if assigned == 0:
        raise RuntimeError(
            "No pose_init joints were matched to FK joint order. "
            "Please check pose_init joint names against robot model joints."
        )

    qpos_fk = np.concatenate(
        [
            np.asarray(root_pos, dtype=np.float64).reshape(3),
            _normalize_quat_wxyz(np.asarray(root_rot, dtype=np.float64).reshape(4)),
            dof_vec,
        ],
        axis=0,
    )

    robot_centers, robot_Rs = fk_solver.fk(qpos_fk)
    robot_link_indices = _to_body_name_to_index(fk_solver.body_names)

    used_robot_links = set()
    ik_match_table1 = {}
    ik_match_table2 = {}
    skipped = []

    for entry in mappings:
        robot_link = entry.get("robot_link", None)
        human_bone = entry.get("human_bone", None)
        if not robot_link or not human_bone:
            raise ValueError(f"Invalid mapping entry (missing robot_link/human_bone): {entry}")

        if robot_link in used_robot_links:
            raise ValueError(f"Duplicate robot_link '{robot_link}' in mappings.")

        if robot_link not in robot_link_indices:
            msg = f"robot_link '{robot_link}' not found in robot FK bodies"
            if args.allow_missing:
                skipped.append(msg)
                continue
            raise KeyError(msg)

        if human_bone not in human_frame:
            msg = f"human_bone '{human_bone}' not found in loaded {args.src_human} frame"
            if args.allow_missing:
                skipped.append(msg)
                continue
            raise KeyError(msg)

        human_quat = _normalize_quat_wxyz(np.array(human_frame[human_bone][1], dtype=np.float64))
        human_rot = R.from_quat(human_quat, scalar_first=True)

        robot_idx = robot_link_indices[robot_link]
        robot_rot = R.from_matrix(robot_Rs[robot_idx])

        rot_offset = human_rot.inv() * robot_rot

        # Optional per-mapping extra/override (higher priority than global root config).
        quat_extra = _maybe_get_quat(entry, ["quat_extra_wxyz", "quat_extra"])
        quat_override = _maybe_get_quat(entry, ["quat_offset_override_wxyz", "quat_offset_override"])

        if quat_extra is None and robot_link == robot_root_name and root_quat_extra is not None:
            quat_extra = root_quat_extra
        if quat_override is None and robot_link == robot_root_name and root_quat_override is not None:
            quat_override = root_quat_override

        if quat_extra is not None:
            rot_offset = R.from_quat(quat_extra, scalar_first=True) * rot_offset

        if quat_override is not None:
            rot_offset_quat = quat_override.tolist()
        else:
            rot_offset_quat = _normalize_quat_wxyz(rot_offset.as_quat(scalar_first=True)).tolist()

        t1_pos, t1_rot, t2_pos, t2_rot = _extract_weights(entry)
        t1_pos_offset, t2_pos_offset = _extract_pos_offsets(entry)

        ik_match_table1[robot_link] = [human_bone, t1_pos, t1_rot, t1_pos_offset, rot_offset_quat]
        ik_match_table2[robot_link] = [human_bone, t2_pos, t2_rot, t2_pos_offset, rot_offset_quat]
        used_robot_links.add(robot_link)

    if len(ik_match_table1) == 0:
        raise RuntimeError("No valid mapping entry remained after validation.")

    human_scale_table = simple_cfg.get("human_scale_table", None)
    human_scale_table_xsens = simple_cfg.get("human_scale_table_xsens", None)
    if human_scale_table is None:
        human_scale_table = _build_default_human_scale_table(mappings, human_root_name)
    else:
        human_scale_table = {k: float(v) for k, v in human_scale_table.items()}
    # Keep scale table consistent with mapped human bones to avoid runtime KeyError.
    mapped_human_bones = {v[0] for v in ik_match_table1.values()}
    mapped_human_bones.add(human_root_name)
    human_scale_table = {k: v for k, v in human_scale_table.items() if k in mapped_human_bones}
    for bone_name in mapped_human_bones:
        if bone_name not in human_scale_table:
            human_scale_table[bone_name] = 1.0

    output_cfg = {
        "robot_root_name": robot_root_name,
        "human_root_name": human_root_name,
        "ground_height": float(simple_cfg.get("ground_height", 0.0)),
        "human_height_assumption": float(simple_cfg.get("human_height_assumption", actual_human_height)),
        "use_ik_match_table1": bool(simple_cfg.get("use_ik_match_table1", True)),
        "use_ik_match_table2": bool(simple_cfg.get("use_ik_match_table2", True)),
        "human_scale_table": human_scale_table,
        "ik_match_table1": ik_match_table1,
        "ik_match_table2": ik_match_table2,
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_cfg, f, indent=2, ensure_ascii=False)

    print("[SUCCESS] IK config generated:")
    print(f"  - output: {out_path}")
    print(f"  - src_human: {args.src_human}")
    print(f"  - robot_name: {args.robot_name}")
    print(f"  - robot_xml: {robot_xml}")
    print(f"  - mapped links: {len(ik_match_table1)}")
    print(f"  - assigned pose_init joints: {assigned}/{len(joints_dict)}")
    if skipped:
        print(f"  - skipped entries: {len(skipped)}")
        for msg in skipped:
            print(f"    * {msg}")

    if args.src_human == "smplx":
        xrobot_cfg = _build_xrobot_cfg_from_smplx_cfg(output_cfg)
        xrobot_out_path = _derive_xrobot_output_path(out_path)
        xrobot_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(xrobot_out_path, "w", encoding="utf-8") as f:
            json.dump(xrobot_cfg, f, indent=2, ensure_ascii=False)
        print(f"  - xrobot output: {xrobot_out_path}")

    if args.src_human == "bvh":
        xsens_cfg = _build_xsens_cfg_from_bvh_cfg(output_cfg, human_scale_table_xsens)
        xsens_out_path = _derive_bvh_xsens_output_path(out_path)
        xsens_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(xsens_out_path, "w", encoding="utf-8") as f:
            json.dump(xsens_cfg, f, indent=2, ensure_ascii=False)
        print(f"  - xsens output: {xsens_out_path}")

    if args.preview:
        print("[INFO] Launching T-pose axis preview...")
        _preview_tpose_axes(
            robot_xml=robot_xml,
            qpos_fk=qpos_fk,
            human_frame=human_frame,
            human_scale_table=human_scale_table,
            ground_height=float(simple_cfg.get("ground_height", 0.0)),
            human_root_name=human_root_name,
            robot_root_name=robot_root_name,
            ik_match_table1=ik_match_table1,
            preview_seconds=float(args.preview_seconds),
            axis_scale=float(args.preview_axis_scale),
            show_human_raw_axes=bool(args.preview_show_human_raw_axes),
            show_corrected_axes=bool(args.preview_show_corrected_axes),
        )


if __name__ == "__main__":
    main()
