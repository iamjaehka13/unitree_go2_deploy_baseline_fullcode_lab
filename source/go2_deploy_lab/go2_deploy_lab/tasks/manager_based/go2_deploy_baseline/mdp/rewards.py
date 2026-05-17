"""Reward terms matching the legacy Go2 deploy baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_deploy(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    body_speed_threshold: float = 0.3,
    stand_still_scale: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation more strongly while standing."""
    asset = env.scene[asset_cfg.name]
    joint_error = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    cmd_speed = torch.linalg.norm(command[:, :2], dim=1)
    body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    moving = torch.logical_or(cmd_speed > 0.0, body_speed > body_speed_threshold)
    return torch.where(moving, joint_error, stand_still_scale * joint_error)


def air_time_variance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Penalize asymmetric foot air times."""
    if hasattr(env, "_deploy_feet_air_time"):
        air_time = torch.clamp(env._deploy_feet_air_time, max=0.5)
        return torch.var(air_time, dim=1)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = torch.clamp(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids], max=0.5)
    return torch.var(air_time, dim=1)


def feet_air_time_deploy(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Gym deploy foot air-time reward."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not hasattr(env, "_deploy_feet_air_time"):
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward

    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    contact_filt = torch.logical_or(contact, env._deploy_last_contacts)
    env._deploy_last_contacts[:] = contact
    first_contact = torch.logical_and(env._deploy_feet_air_time > 0.0, contact_filt)
    env._deploy_feet_air_time += env.step_dt
    reward = torch.sum((env._deploy_feet_air_time - threshold) * first_contact, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    env._deploy_feet_air_time *= torch.logical_not(contact_filt)
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot motion while a foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold).float()
    asset = env.scene[asset_cfg.name]
    foot_horiz_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    return torch.sum(foot_horiz_speed * contact, dim=1)


def undesired_contacts_current(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count undesired current-step contacts, matching legged_gym's collision reward."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.sum(torch.linalg.norm(net_forces, dim=-1) > threshold, dim=1)


def foot_gait(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    gait_period: float = 0.6,
    gait_stance_ratio: float = 0.56,
    gait_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0),
    command_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Reward contacts that match the desired trot phase."""
    command = env.command_manager.get_command(command_name)
    cmd_active = (torch.linalg.norm(command[:, :2], dim=1) >= command_threshold).float()
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, gait_period) / gait_period
    offsets = torch.tensor(gait_offsets, dtype=torch.float, device=env.device)
    foot_phase = torch.remainder(phase.unsqueeze(1) + offsets.unsqueeze(0), 1.0)
    desired_stance = foot_phase < gait_stance_ratio
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    gait_match = torch.logical_not(torch.logical_xor(contact, desired_stance)).float()
    return torch.mean(gait_match, dim=1) * cmd_active


def termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminal penalty, excluding time-outs."""
    return torch.logical_and(env.reset_terminated, torch.logical_not(env.reset_time_outs)).float()
