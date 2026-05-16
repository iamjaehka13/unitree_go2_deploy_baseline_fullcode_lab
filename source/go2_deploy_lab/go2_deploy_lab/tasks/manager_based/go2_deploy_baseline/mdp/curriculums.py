"""Curriculum terms for the stateful Go2 deploy migration."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def deploy_command_curriculum(
    env,
    env_ids: Sequence[int],
    tracking_lin_reward_name: str = "tracking_lin_vel",
    tracking_ang_reward_name: str = "tracking_ang_vel",
    command_name: str = "base_velocity",
    action_name: str = "joint_pos",
):
    """Update gym deploy curriculum state from completed episode statistics."""
    if isinstance(env_ids, slice) or len(env_ids) == 0:
        return _deploy_curriculum_state(env)

    episode_durations = torch.clamp(env.episode_length_buf[env_ids].float() * env.step_dt, min=1e-6)
    track_lin_scale = max(float(getattr(env.cfg, "deploy_tracking_lin_vel_scale", 3.5)), 1e-6)
    track_ang_scale = max(float(getattr(env.cfg, "deploy_tracking_ang_vel_scale", 1.75)), 1e-6)

    track_lin_sum = env.reward_manager._episode_sums[tracking_lin_reward_name][env_ids]
    track_ang_sum = env.reward_manager._episode_sums[tracking_ang_reward_name][env_ids]
    track_lin = track_lin_sum / episode_durations / track_lin_scale
    track_ang = track_ang_sum / episode_durations / track_ang_scale
    tracking = torch.mean(0.5 * (track_lin + track_ang)).item()
    timeout_rate = torch.mean(env.reset_time_outs[env_ids].float()).item()
    fall_rate = torch.mean(torch.logical_not(env.reset_time_outs[env_ids]).float()).item()

    alpha = float(env.cfg.deploy_curriculum_ema_alpha)
    env._deploy_tracking_ema = (1.0 - alpha) * env._deploy_tracking_ema + alpha * tracking
    env._deploy_timeout_rate_ema = (1.0 - alpha) * env._deploy_timeout_rate_ema + alpha * timeout_rate
    env._deploy_fall_rate_ema = (1.0 - alpha) * env._deploy_fall_rate_ema + alpha * fall_rate

    if env._deploy_curriculum_cooldown > 0:
        env._deploy_curriculum_cooldown -= 1
        _sync_deploy_terms(env, command_name, action_name)
        return _deploy_curriculum_state(env)

    ready = (
        env._deploy_timeout_rate_ema >= float(env.cfg.deploy_curriculum_ready_timeout_rate)
        and env._deploy_tracking_ema >= float(env.cfg.deploy_curriculum_ready_tracking)
        and env._deploy_fall_rate_ema <= float(env.cfg.deploy_curriculum_ready_fall_rate)
    )
    hard = env._deploy_fall_rate_ema >= float(env.cfg.deploy_curriculum_hard_fall_rate)

    env._deploy_ready_streak = env._deploy_ready_streak + 1 if ready else 0
    env._deploy_hard_streak = env._deploy_hard_streak + 1 if hard else 0

    if env._deploy_hard_streak >= int(env.cfg.deploy_curriculum_hard_streak):
        env._deploy_curriculum_level = max(
            float(env.cfg.deploy_curriculum_level_min),
            env._deploy_curriculum_level - float(env.cfg.deploy_curriculum_step_down),
        )
        env._deploy_hard_streak = 0
        env._deploy_ready_streak = 0
        env._deploy_curriculum_cooldown = int(env.cfg.deploy_curriculum_cooldown)
    elif env._deploy_ready_streak >= int(env.cfg.deploy_curriculum_ready_streak):
        env._deploy_curriculum_level = min(
            float(env.cfg.deploy_curriculum_level_max),
            env._deploy_curriculum_level + float(env.cfg.deploy_curriculum_step_up),
        )
        env._deploy_hard_streak = 0
        env._deploy_ready_streak = 0
        env._deploy_curriculum_cooldown = int(env.cfg.deploy_curriculum_cooldown)

    _refresh_curriculum_targets(env)
    _sync_deploy_terms(env, command_name, action_name)
    return _deploy_curriculum_state(env)


def _refresh_curriculum_targets(env):
    env._deploy_curriculum_level = float(
        min(
            max(env._deploy_curriculum_level, float(env.cfg.deploy_curriculum_level_min)),
            float(env.cfg.deploy_curriculum_level_max),
        )
    )
    env._deploy_obs_noise_level_cur = env._deploy_curriculum_level * float(env.cfg.deploy_obs_noise_level_max)
    env._deploy_action_noise_std_cur = env._deploy_curriculum_level * float(env.cfg.deploy_action_noise_std_max)
    delay_easy = int(env.cfg.deploy_delay_easy_max_steps)
    delay_hard = int(env.cfg.deploy_delay_max_steps)
    env._deploy_delay_max_cur = int(round(delay_easy + (delay_hard - delay_easy) * env._deploy_curriculum_level))


def _sync_deploy_terms(env, command_name: str, action_name: str):
    command_term = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    command_term.set_curriculum_level(env._deploy_curriculum_level)
    action_term.set_curriculum_level(env._deploy_curriculum_level)


def _deploy_curriculum_state(env) -> dict[str, float]:
    return {
        "level": float(env._deploy_curriculum_level),
        "tracking_ema": float(env._deploy_tracking_ema),
        "timeout_rate_ema": float(env._deploy_timeout_rate_ema),
        "fall_rate_ema": float(env._deploy_fall_rate_ema),
        "delay_max_cur": float(env._deploy_delay_max_cur),
        "action_noise_std": float(env._deploy_action_noise_std_cur),
        "obs_noise_level": float(env._deploy_obs_noise_level_cur),
    }
