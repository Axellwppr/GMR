from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    ext = os.path.splitext(robot_motion_path)[1].lower()
    if ext == ".pkl":
        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    elif ext == ".npz":
        # Load NPZ exported by scripts/smplx_to_robot_dataset.py
        npz = dict(np.load(robot_motion_path, allow_pickle=True))
        motion_data = npz
        # Ensure scalar fps
        fps_val = npz.get("fps", 30)
        motion_fps = float(fps_val) if np.isscalar(fps_val) or np.array(fps_val).shape == () else float(np.array(fps_val).item())
        motion_root_pos = np.array(npz["root_pos"])  # (T,3)
        # Convert XYZW (saved) -> WXYZ for Mujoco
        root_xyzw = np.array(npz["root_rot"])  # (T,4)
        motion_root_rot = root_xyzw[:, [3, 0, 1, 2]]
        motion_dof_pos = np.array(npz["dof_pos"])  # (T,D)
        motion_local_body_pos = np.array(npz.get("local_body_pos", []))
        # Use body_names as link_body_list equivalent if present
        motion_link_body_list = npz.get("body_names", [])
    else:
        raise ValueError(f"Unsupported motion file extension: {ext}. Expected .pkl or .npz")
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True, follow_camera=False)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()
