# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
from typing import Dict

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.
    
    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.
    
    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()
    
    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]
    
    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)
    
    # Interpolate translations
    interp_translations = np.stack([
        np.interp(target_times, times, translations[:, i])
        for i in range(3)
    ], axis=-1)
    
    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()
    
    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations
    
    return torch.tensor(interp_poses, dtype=torch.float32, device=device)

def look_at_rotation(direction: torch.Tensor, up: torch.Tensor = torch.tensor([0., 0., 1.])) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix

def get_interp_novel_trajectories(
    dataset_type: str,
    # scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: dict,
    # traj_type: str = "front_center_interp",
    target_frames: int = 100,
    cam2ego: Dict[int, torch.Tensor] = None,
    ego2worlds: Dict[int, torch.Tensor] = None,
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]
    
    trajectory_generators = {
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory
    }
    if traj_type.type in trajectory_generators:
        return trajectory_generators[traj_type.type](dataset_type, per_cam_poses, original_frames, target_frames)

    custom_trajectory_generators = {
        "relative_lane_shift": relative_lane_shift,
        "absolute_poses": absolute_poses,
        "lane_change_smooth_trajectory": lane_change_smooth_trajectory,
    }

    if traj_type.type in custom_trajectory_generators:
        return custom_trajectory_generators[traj_type.type](cam2ego, ego2worlds, traj_type)

    raise ValueError(f"Unknown trajectory type: {traj_type}")

def front_center_interp(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int, num_loops: int = 1
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][::original_frames//4]  # Select every 4th frame as key frame
    return interpolate_poses(key_poses, target_frames)

def s_curve(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(cam in per_cam_poses.keys() for cam in [0, 1, 2]), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat([
        per_cam_poses[0][0:1],
        per_cam_poses[1][original_frames//4:original_frames//4+1],
        per_cam_poses[0][original_frames//2:original_frames//2+1],
        per_cam_poses[2][3*original_frames//4:3*original_frames//4+1],
        per_cam_poses[0][-1:]
    ], dim=0)
    return interpolate_poses(key_poses, target_frames)

def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert 1 in per_cam_poses.keys() or 2 in per_cam_poses.keys(), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp([0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()]))
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(interpolated_rotation, device=start_pose.device)
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)



def relative_lane_shift(
    cam2egos: Dict[int, torch.Tensor] = None, ego2worlds: Dict[int, torch.Tensor] = None, traj_type: dict = None
) -> torch.Tensor:
    """
    Generate a trajectory by shifting the ego vehicle along the x-axis.
    Currently only support x-axis shift.
    """
    dis_shift = traj_type.distance

    cam_poses = []
    for cam_id in cam2egos:
        cam2ego = cam2egos[cam_id].clone().detach()
        # cam2ego[0, 3] += dis_shift  # X-Front
        cam2ego[1, 3] += dis_shift  # Y-Left
        # cam2ego[2, 3] += dis_shift  # Z-Up
        for ego2world in ego2worlds[cam_id]:
            cam_pose = ego2world @ cam2ego
            cam_poses.append(cam_pose)

    return torch.stack(cam_poses)


def lane_change_smooth_trajectory(
    cam2egos: Dict[int, torch.Tensor],
    ego2worlds: Dict[int, torch.Tensor],
    traj_type: dict
) -> torch.Tensor:

    direction = traj_type.get("direction", "x")
    total_shift = traj_type.get("distance", 3.0)
    shift_time = traj_type.get("shift_secs", 5)
    smooth_frames = shift_time * 10
    #interp_mode = traj_type.get("mode", "linear")

    all_poses = []

    for cam_id in cam2egos:
        base_cam2ego = cam2egos[cam_id].clone().detach()
        num_required = len(ego2worlds[cam_id])

        cam_traj = []
        for i in range(num_required):
            if i < smooth_frames:
                t = i / (smooth_frames - 1)
                shift = total_shift * t
            else:
                shift = total_shift

            shifted_cam2ego = base_cam2ego.clone()
            shifted_cam2ego[1, 3] += shift # 平移

            base_ego2world = ego2worlds[cam_id][i]
            cam_pose = base_ego2world @ shifted_cam2ego
            cam_traj.append(cam_pose)

        all_poses.extend(cam_traj)

    return torch.stack(all_poses)


def absolute_poses(
    cam2egos: Dict[int, torch.Tensor] = None, ego2worlds: Dict[int, torch.Tensor] = None, traj_type: dict = None
) -> torch.Tensor:
    raise NotImplementedError("Absolute trajectory generation is not implemented yet.")

