"""Termination terms matching the legacy Go2 deploy baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out_after_max_length(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Match legged_gym's timeout boundary: terminate only after max length is exceeded."""
    return env.episode_length_buf > env.max_episode_length


def illegal_contact_current(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate on current-step contact force, matching legged_gym contact checks."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.linalg.norm(net_forces, dim=-1) > threshold, dim=1)


def bad_roll_pitch(
    env: ManagerBasedRLEnv,
    roll_limit: float = 0.8,
    pitch_limit: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on excessive base roll or pitch."""
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.logical_or(torch.abs(roll) > roll_limit, torch.abs(pitch) > pitch_limit)
