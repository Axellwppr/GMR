import argparse
import json
import pathlib
import os
import multiprocessing as mp

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import IK_CONFIG_ROOT
import gc
import time
import psutil
import tracemalloc

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import PchipInterpolator

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
    rot_src = R.from_quat(q_xyzw_src)   # SciPy expects XYZW
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

def check_memory(threshold_gb=5):  # adjust based on your available memory
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent

# ------------------------
# Per-process global state
# ------------------------
_GLOBAL_STATE = {
    "initialized": False,
    "tgt_robot": None,
    "retargeter": None,
    "kinematics_model": None,
    "device": None,
    "smplx_folder": None,
    "body_model_cache": {},  # gender -> smplx model
}

def _get_body_model(gender: str):
    import smplx
    cache = _GLOBAL_STATE["body_model_cache"]
    if gender not in cache:
        cache[gender] = smplx.create(
            _GLOBAL_STATE["smplx_folder"],
            "smplx",
            gender=str(gender),
            use_pca=False,
        )
    return cache[gender]

def _init_worker(tgt_robot, smplx_folder, device="cuda:0", verbose_retargeter=False):
    # Initialize heavy components once per worker
    _GLOBAL_STATE["tgt_robot"] = tgt_robot
    _GLOBAL_STATE["smplx_folder"] = str(smplx_folder)
    _GLOBAL_STATE["device"] = device
    _GLOBAL_STATE["retargeter"] = GMR(
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=None,
        verbose=verbose_retargeter,
    )
    _GLOBAL_STATE["kinematics_model"] = KinematicsModel(_GLOBAL_STATE["retargeter"].xml_file, device=device)
    _GLOBAL_STATE["initialized"] = True



def process_file(smplx_file_path, tgt_file_path, tgt_robot, SMPLX_FOLDER, tgt_folder, total_files, counter, verbose=False):
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")

    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()

    # Initial checks (with optional logging)
    log_memory("Initial memory usage")

    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(15)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return

    # Lazy per-worker init (for direct calls without pool initializer)
    if not _GLOBAL_STATE["initialized"]:
        _init_worker(tgt_robot, SMPLX_FOLDER)

    # Load SMPL-X sequence and run the body model (reuse cached models by gender)
    smplx_data = dict(np.load(smplx_file_path, allow_pickle=True))

    ### interx
    # smplx_data["mocap_frame_rate"] = np.float32(120)
    # betas = np.zeros(16)
    # betas[:10] = smplx_data["betas"].reshape(-1)
    # smplx_data["betas"] = betas

    # trans = smplx_data["trans"]
    # root_aa = smplx_data['root_orient'][:, :3]
    # root_aa = R.from_rotvec(root_aa)
    # local_rot = R.from_matrix([
    #     [1, 0, 0],
    #     [0, 0, -1],
    #     [0, 1, 0]
    # ])
    # trans = local_rot.apply(trans)
    # root_aa = (local_rot * root_aa).as_rotvec()
    # smplx_data['trans'][:] = trans
    # smplx_data['root_orient'][:] = root_aa
    ### interx

    gender = str(smplx_data["gender"]) if "gender" in smplx_data else "neutral"
    body_model = _get_body_model(gender)

    if len(smplx_data["betas"].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    tgt_fps = 10
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, tgt_fps=tgt_fps)
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return

    # retarget using a persistent per-worker retargeter
    retargeter = _GLOBAL_STATE["retargeter"]
    retargeter.set_human_height(human_height)
    retargeter.reset_configuration()
    qpos_list = []
    for smplx_frame_data in smplx_frame_data_list:
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())

    qpos_list = np.array(qpos_list)

    log_memory("After retargeting")

    # -----------------------------
    # 10 Hz -> 50 Hz 重采样（平滑）
    # -----------------------------
    # 拆分 10 Hz 的根位姿与 DoF
    root_pos_10 = qpos_list[:, :3]
    root_rot_10 = qpos_list[:, 3:7].copy()
    # 将四元数从 (w,x,y,z) 旋转成 (x,y,z,w) 以适配 SciPy/KinematicsModel
    root_quat_xyzw_10 = root_rot_10.copy()
    root_quat_xyzw_10[:, [0, 1, 2, 3]] = root_quat_xyzw_10[:, [1, 2, 3, 0]]
    dof_pos_10 = qpos_list[:, 7:]

    src_fps_for_resample = aligned_fps  # 这里通常是 10
    tgt_fps_after = 50

    # ### NEW: 使用 SLERP + PCHIP 重采样到 50 Hz
    root_pos, root_quat_xyzw, dof_pos, export_fps = resample_motion_to_fps(
        root_pos_10, root_quat_xyzw_10, dof_pos_10, src_fps_for_resample, tgt_fps_after, unwrap_angles=True
    )

    device = _GLOBAL_STATE["device"]
    kinematics_model = _GLOBAL_STATE["kinematics_model"]

    num_frames = root_pos.shape[0]

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, local_body_rot = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    log_memory("After forward kinematics")

    body_names = kinematics_model.body_names
    joint_names = kinematics_model.joint_names

    HEIGHT_ADJUST = True
    if HEIGHT_ADJUST:
        # height adjust to ensure the lowerset part is on the ground
        body_pos, _ = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(root_quat_xyzw).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
        ground_offset = 0.0
        lowerst_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground

    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]

    motion_data = {
        "fps": np.array(export_fps, dtype=np.float32),   # 标量也转成 ndarray
        "root_pos": root_pos,
        "root_rot": root_quat_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "local_body_rot": local_body_rot.detach().cpu().numpy(),
        "body_names": np.array(body_names),  # list[str] 转成 ndarray(dtype=object 或 str)
        "joint_names": np.array(joint_names),  # list[str] 转成 ndarray(dtype=object 或 str)
    }

    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    np.savez_compressed(tgt_file_path, **motion_data)

    # pkl
    motion_data = {
        "fps": export_fps,
        "root_pos": root_pos,
        "root_rot": root_quat_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
    }

    os.makedirs(os.path.dirname(tgt_file_path.replace(".npz", ".pkl")), exist_ok=True)
    with open(tgt_file_path.replace(".npz", ".pkl"), "wb") as f:
        pickle.dump(motion_data, f)

    counter.value += 1
    done = counter.value
    print(f"Processed {done}/{total_files}: {tgt_file_path}")

    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)

        tracemalloc.stop()

    # clean cache
    torch.cuda.empty_cache()
    gc.collect()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
                        )
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=6, type=int)
    args = parser.parse_args()
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions_paths = [hard_motions_folder / "0.txt",
                          hard_motions_folder / "1.txt"]
    # Collect motion identifiers that may be either basenames or include the
    # last two directory levels plus filename (without extension).
    hard_motion_keys = set()
    for hard_motions_path in hard_motions_paths:
        with open(hard_motions_path, "r") as f:
            for line in f:
                if "Motion:" not in line:
                    continue
                motion_path = line.split(":", 1)[1]
                # Drop trailing difficulty, whitespace and extension
                motion_path = motion_path.split(",", 1)[0].strip()
                motion_no_ext = os.path.splitext(motion_path)[0]
                # Normalize to POSIX-like separators for consistent comparison
                motion_no_ext = motion_no_ext.replace("\\", "/")
                # Key 1: basename without extension
                hard_motion_keys.add(os.path.basename(motion_no_ext))
                # Key 2: last two parent directories + filename (3 components)
                parts = motion_no_ext.split("/")
                if len(parts) >= 3:
                    hard_motion_keys.add("/".join(parts[-3:]))
                
                
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder)
                if not os.path.exists(tgt_file_path) or args.override:
                    args_list.append((smplx_file_path, tgt_file_path, args.robot, SMPLX_FOLDER, tgt_folder))
    print("full args_list:", len(args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    
    new_args_list = []
    for arguments in args_list:
        smplx_path = arguments[0]
        # Build comparable keys from the source file path relative to src_folder
        rel_path = os.path.relpath(smplx_path, src_folder)
        rel_path = rel_path.replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_path)[0]
        base_no_ext = os.path.basename(rel_no_ext)

        # Candidate keys to match against hard motions
        parts = rel_no_ext.split("/")
        last3 = "_".join(parts[-3:]) if len(parts) >= 3 else base_no_ext
        # Skip if this motion is marked hard (by basename or last two dirs + name)
        if base_no_ext in hard_motion_keys or last3 in hard_motion_keys:
            continue

        # Also skip motions by excluded content in basename
        if any(content in last3 for content in exclude_file_content):
            continue
        new_args_list.append(arguments)
    args_list = new_args_list
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")


    # 新增：全局计数器（用Manager以支持进程间共享）
    manager = mp.Manager()
    counter = manager.Value('i', 0)

    # Initialize a worker pool with persistent per-process state
    with mp.Pool(args.num_cpus, initializer=_init_worker, initargs=(args.robot, SMPLX_FOLDER, "cpu", False)) as pool:
        pool.starmap(process_file, [args + (total_files, counter, verbose) for args in args_list])
    
    # _init_worker(args.robot, SMPLX_FOLDER, "cpu", False)
    # process_file(*(args_list[0] + (total_files, counter, verbose)))

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
