"""Observation terms that match the low-level Go2 deploy policy interface."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the previous raw policy action, matching the deploy observation contract."""
    return applied_action(env)


def applied_action(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the previous delayed/noisy action applied to the PD target."""
    action_term = env.action_manager.get_term(action_name)
    if hasattr(action_term, "last_applied_raw_actions"):
        return action_term.last_applied_raw_actions
    return env.action_manager.prev_action


def gait_phase(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    gait_period: float = 0.6,
    command_threshold: float = 0.1,
    stand_phase_lock: bool = True,
) -> torch.Tensor:
    """Return sin/cos gait phase, with optional phase lock while standing."""
    if hasattr(env, "_deploy_phase") and gait_period == 0.6:
        phase = env._deploy_phase
    else:
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


def deploy_actor_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    action_name: str = "joint_pos",
    gait_period: float = 0.6,
    command_threshold: float = 0.1,
    stand_phase_lock: bool = True,
    add_noise: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the exact 47-dim gym deploy actor observation."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    phase = gait_phase(
        env,
        command_name=command_name,
        gait_period=gait_period,
        command_threshold=command_threshold,
        stand_phase_lock=stand_phase_lock,
    )
    obs = torch.cat(
        (
            asset.data.root_ang_vel_b * 0.25,
            asset.data.projected_gravity_b,
            command[:, :3] * torch.tensor((2.0, 2.0, 0.25), device=env.device),
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
            asset.data.joint_vel[:, asset_cfg.joint_ids] * 0.05,
            applied_action(env, action_name=action_name),
            phase,
        ),
        dim=-1,
    )

    if add_noise:
        obs_noise_level = float(getattr(env, "_deploy_obs_noise_level_cur", 0.0))
        if obs_noise_level > 0.0:
            obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * _deploy_noise_scale_vec(obs, env) * obs_noise_level
    return obs


def _deploy_noise_scale_vec(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    noise_vec = torch.zeros_like(obs)
    noise_vec[:, 0:3] = 0.2 * 0.25
    noise_vec[:, 3:6] = 0.05
    noise_vec[:, 9:21] = 0.01
    noise_vec[:, 21:33] = 1.5 * 0.05
    return noise_vec


def feet_pos_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """Return foot heights used by the asymmetric critic."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2]


def base_lin_vel_scaled(
    env: ManagerBasedRLEnv,
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return scaled base linear velocity before final observation clipping."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b * scale


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Return the current foot air-time buffer."""
    if hasattr(env, "_deploy_feet_air_time"):
        return env._deploy_feet_air_time
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]


def foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return binary foot contacts from vertical contact force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold).float()


def contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Return flattened xyz contact forces for the selected bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].reshape(env.num_envs, -1)


def constant(env: ManagerBasedRLEnv, value: float, dim: int) -> torch.Tensor:
    """Return a constant observation term for deploy metadata not stored by Isaac Lab managers."""
    return torch.full((env.num_envs, dim), value, device=env.device)


def deploy_friction_coeffs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._deploy_friction_coeffs_cur


def deploy_delay_steps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return action_term.delay_steps.float().unsqueeze(1)


def deploy_action_noise_std(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return torch.full((env.num_envs, 1), float(action_term.action_noise_std), device=env.device)


def deploy_obs_noise_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.full((env.num_envs, 1), float(env._deploy_obs_noise_level_cur), device=env.device)


def deploy_push_history_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._deploy_push_history_xy
