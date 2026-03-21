import argparse
import pathlib
import os
import csv
from types import SimpleNamespace
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.xsens import load_xsens_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import PchipInterpolator


def load_100style_frame_cuts(csv_path):
    """
    Load 100STYLE frame cuts from CSV.
    Returns: dict[style_name][code] = (start_frame_1based, stop_frame_1based)
    """
    cuts = {}
    if not os.path.exists(csv_path):
        return cuts

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            style = row.get("STYLE_NAME", "").strip()
            if not style:
                continue
            style_cuts = {}
            for key, value in row.items():
                if not key or not key.endswith("_START"):
                    continue
                code = key[:-6]  # remove "_START"
                stop_key = f"{code}_STOP"
                start_val = (value or "").strip()
                stop_val = (row.get(stop_key) or "").strip()
                if not start_val or not stop_val:
                    continue
                if start_val.upper() == "N/A" or stop_val.upper() == "N/A":
                    continue
                try:
                    style_cuts[code] = (int(start_val), int(stop_val))
                except ValueError:
                    continue
            cuts[style] = style_cuts
    return cuts


def get_100style_cut_for_file(bvh_file_path, cuts):
    """
    Infer 100STYLE (style, code) from filename.
    Example:
      Angry_BR.bvh -> style=Angry, code=BR
      Angry_BR_02.bvh -> style=Angry, code=BR
    Returns:
      (start_0based, end_exclusive) or (None, None)
    """
    p = pathlib.Path(bvh_file_path)
    style = p.parent.name
    stem = p.stem
    prefix = f"{style}_"
    if not stem.startswith(prefix):
        return None, None

    suffix = stem[len(prefix):]
    code = suffix.split("_")[0].strip()
    if not code:
        return None, None

    style_cuts = cuts.get(style)
    if not style_cuts or code not in style_cuts:
        return None, None

    start_1based, stop_1based = style_cuts[code]
    # CSV uses frame indices in 1-based inclusive convention.
    start_0based = max(start_1based - 1, 0)
    end_exclusive = max(stop_1based, start_0based + 1)
    return start_0based, end_exclusive


def _resample_vec_series(times_src, X_src, times_tgt):
    """
    PCHIP per-dim for vectors. X_src: (N, D) -> (M, D)
    """
    N = X_src.shape[0]
    if N == 1:
        return np.repeat(X_src, len(times_tgt), axis=0)
    outs = []
    for k in range(X_src.shape[1]):
        f = PchipInterpolator(times_src, X_src[:, k])
        outs.append(f(times_tgt))
    return np.stack(outs, axis=1)


def _resample_quat_series_xyzw(times_src, q_xyzw_src, times_tgt):
    """
    Quaternion SLERP using SciPy. q in XYZW (scalar-last).
    """
    N = q_xyzw_src.shape[0]
    if N == 1:
        return np.repeat(q_xyzw_src, len(times_tgt), axis=0)
    rot_src = R.from_quat(q_xyzw_src)
    slerp = Slerp(times_src, rot_src)
    rot_tgt = slerp(times_tgt)
    return rot_tgt.as_quat()


def resample_motion_to_fps(root_pos, root_quat_xyzw, dof_pos, src_fps, tgt_fps, unwrap_angles=True):
    """
    root_pos: (T,3), root_quat_xyzw: (T,4) [x,y,z,w], dof_pos: (T,D)
    Returns resampled (root_pos, root_quat_xyzw, dof_pos) and fps=tgt_fps
    """
    T = root_pos.shape[0]
    duration = (T - 1) / float(src_fps) if T > 1 else 0.0
    M = int(np.round(duration * float(tgt_fps))) + 1 if duration > 0 else 1
    times_src = np.linspace(0.0, duration, T)
    times_tgt = np.linspace(0.0, duration, M)

    dof_src = np.unwrap(dof_pos, axis=0) if unwrap_angles and T > 1 else dof_pos

    root_pos_50 = _resample_vec_series(times_src, root_pos, times_tgt)
    root_quat_xyzw_50 = _resample_quat_series_xyzw(times_src, root_quat_xyzw, times_tgt)
    dof_pos_50 = _resample_vec_series(times_src, dof_src, times_tgt)

    return root_pos_50, root_quat_xyzw_50, dof_pos_50, tgt_fps


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "xsens", "bvh_xsens"],
        default="lafan1",
        help="BVH skeleton format.",
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    format_key = "xsens" if args.format == "bvh_xsens" else args.format
    frame_cuts_csv = "/home/axell/Desktop/dataset_new/raw/100STYLE/Frame_Cuts.csv"
    cuts_100style = load_100style_frame_cuts(frame_cuts_csv) if format_key == "xsens" else {}

   
   
        
    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            
            # get the target file path
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".npz")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            # Load LAFAN1 trajectory
            try:
                if format_key == "xsens":
                    cut_start, cut_end = get_100style_cut_for_file(bvh_file_path, cuts_100style)
                    xsens_args = SimpleNamespace(
                        bvh_file=bvh_file_path,
                        scale=0.01,
                        reset_to_zero=False,
                        start=cut_start,
                        end=cut_end,
                        bvh_format="3DSM",
                    )
                    lafan1_data_frames, actual_human_height, frame_time = load_xsens_file(xsens_args)
                    src_fps = int(round(1.0 / frame_time))
                    src_human = "bvh_xsens"
                else:
                    lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file_path, format=format_key)
                    src_fps = 30
                    src_human = f"bvh_{format_key}"
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            
            # Initialize the retargeting system
            retarget = GMR(
                src_human=src_human,
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retarget.xml_file)
            data = mj.MjData(model)

            # 先在 10 Hz 进行 retarget（从 30 Hz 采样到 10 Hz 再 retarget）
            T = len(lafan1_data_frames)
            duration = (T - 1) / float(src_fps) if T > 1 else 0.0
            tgt_retarget_fps = 10
            M10 = int(np.round(duration * float(tgt_retarget_fps))) + 1 if duration > 0 else 1
            times_src = np.linspace(0.0, duration, T)
            times_10 = np.linspace(0.0, duration, M10)
            # 选取最邻近的原始帧来作为 10Hz 的帧
            idx_10 = np.searchsorted(times_src, times_10, side="left")
            idx_10 = np.clip(idx_10, 0, T - 1)
            # 去重，避免重复索引
            idx_10 = np.unique(idx_10)

            qpos_list_10 = []
            for i in idx_10:
                smplx_data = lafan1_data_frames[int(i)]
                qpos = retarget.retarget(smplx_data)
                qpos_list_10.append(qpos.copy())
            qpos_list_10 = np.array(qpos_list_10)

            # Initialize the forward kinematics
            device = "cuda:0"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            
            # 拆分 10 Hz 的根与 DoF，并转换四元数为 XYZW
            root_pos_10 = qpos_list_10[:, :3]
            root_rot_wxyz_10 = qpos_list_10[:, 3:7].copy()
            root_quat_xyzw_10 = root_rot_wxyz_10.copy()
            root_quat_xyzw_10[:, [0, 1, 2, 3]] = root_quat_xyzw_10[:, [1, 2, 3, 0]]
            dof_pos_10 = qpos_list_10[:, 7:]

            # 10 Hz -> 50 Hz 重采样（SLERP + PCHIP）
            root_pos, root_quat_xyzw, dof_pos, export_fps = resample_motion_to_fps(
                root_pos_10, root_quat_xyzw_10, dof_pos_10, src_fps=tgt_retarget_fps, tgt_fps=50, unwrap_angles=True
            )

            num_frames = root_pos.shape[0]
            
            # obtain local body pos
            fk_root_pos = torch.zeros((num_frames, 3), device=device)
            fk_root_rot = torch.zeros((num_frames, 4), device=device)
            fk_root_rot[:, -1] = 1.0
            local_body_pos, local_body_rot = kinematics_model.forward_kinematics(
                fk_root_pos,
                fk_root_rot,
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
            )
            body_names = kinematics_model.body_names
            joint_names = kinematics_model.joint_names

            HEIGHT_ADJUST = True
            if HEIGHT_ADJUST:
                body_pos, _ = kinematics_model.forward_kinematics(
                    torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                    torch.from_numpy(root_quat_xyzw).to(device=device, dtype=torch.float),
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
                )
                ground_offset = 0.0
                lowest_height = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset

            ROOT_ORIGIN_OFFSET = True
            if ROOT_ORIGIN_OFFSET:
                root_pos[:, :2] -= root_pos[0, :2]

            motion_data = {
                "fps": np.array(export_fps, dtype=np.float32),
                "root_pos": root_pos,
                "root_rot": root_quat_xyzw,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos.detach().cpu().numpy(),
                "local_body_rot": local_body_rot.detach().cpu().numpy(),
                "body_names": np.array(body_names),
                "joint_names": np.array(joint_names),
            }

            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            np.savez_compressed(tgt_file_path, **motion_data)

            # pkl
            # motion_data_pkl = {
            #     "fps": export_fps,
            #     "root_pos": root_pos,
            #     "root_rot": root_quat_xyzw,
            #     "dof_pos": dof_pos,
            #     "local_body_pos": local_body_pos.detach().cpu().numpy(),
            #     "link_body_list": body_names,
            # }
            # os.makedirs(os.path.dirname(tgt_file_path.replace(".npz", ".pkl")), exist_ok=True)
            # with open(tgt_file_path.replace(".npz", ".pkl"), "wb") as f:
            #     pickle.dump(motion_data_pkl, f)

    print("Done. saved to ", tgt_folder)
