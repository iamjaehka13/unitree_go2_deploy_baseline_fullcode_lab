"""Termination terms matching the legacy Go2 deploy baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
