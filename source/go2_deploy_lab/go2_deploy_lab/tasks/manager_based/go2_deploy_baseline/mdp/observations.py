"""Observation terms that match the low-level Go2 deploy policy interface."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    gait_period: float = 0.6,
    command_threshold: float = 0.1,
    stand_phase_lock: bool = True,
) -> torch.Tensor:
    """Return sin/cos gait phase, with optional phase lock while standing."""
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, gait_period) / gait_period
    sin_phase = torch.sin(2.0 * math.pi * phase)
    cos_phase = torch.cos(2.0 * math.pi * phase)

    if stand_phase_lock:
        command = env.command_manager.get_command(command_name)
        command_active = torch.logical_or(
            torch.linalg.norm(command[:, :2], dim=1) >= command_threshold,
            torch.abs(command[:, 2]) >= command_threshold,
        )
        sin_phase = torch.where(command_active, sin_phase, torch.zeros_like(sin_phase))
        cos_phase = torch.where(command_active, cos_phase, torch.ones_like(cos_phase))

    return torch.stack((sin_phase, cos_phase), dim=-1)
