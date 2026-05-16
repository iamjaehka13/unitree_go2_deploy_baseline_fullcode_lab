"""Command terms for the stateful Go2 deploy migration."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi


class DeployVelocityCommand(UniformVelocityCommand):
    """Gym deploy command sampler with curriculum interpolation and command post-processing."""

    cfg: "DeployVelocityCommandCfg"

    def __init__(self, cfg: "DeployVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._curriculum_level = float(cfg.curriculum_level_init)

    def set_curriculum_level(self, level: float):
        self._curriculum_level = float(
            min(max(level, float(self.cfg.curriculum_level_min)), float(self.cfg.curriculum_level_max))
        )

    def _resample(self, env_ids: Sequence[int]):
        env_ids = self._as_env_ids(env_ids)
        if len(env_ids) == 0:
            return

        low_s, high_s = self.cfg.resampling_time_range
        low_steps = max(1, int(round(float(low_s) / self._env.step_dt)))
        high_steps = max(low_steps, int(round(float(high_s) / self._env.step_dt)))
        sampled_steps = torch.randint(low_steps, high_steps + 1, (len(env_ids),), device=self.device)
        self.time_left[env_ids] = sampled_steps.float() * self._env.step_dt
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _sample_command_curriculum_levels(self, env_ids: Sequence[int]) -> torch.Tensor:
        t = torch.full((len(env_ids),), self._curriculum_level, dtype=torch.float, device=self.device)
        easy_prob = float(self.cfg.easy_mix_probability)
        if easy_prob > 0.0 and self._curriculum_level > 0.0:
            easy_mask = torch.rand(len(env_ids), device=self.device) < easy_prob
            if easy_mask.any():
                easy_max = min(self._curriculum_level, float(self.cfg.easy_mix_max_level))
                t[easy_mask] = torch.rand(int(easy_mask.sum().item()), device=self.device) * easy_max
        return t

    @staticmethod
    def _lerp_range(low: tuple[float, float], high: tuple[float, float], t: torch.Tensor) -> torch.Tensor:
        low_t = torch.as_tensor(low, dtype=torch.float, device=t.device)
        high_t = torch.as_tensor(high, dtype=torch.float, device=t.device)
        return low_t + (high_t - low_t) * t.unsqueeze(1)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._as_env_ids(env_ids)
        if len(env_ids) == 0:
            return

        t = self._sample_command_curriculum_levels(env_ids)
        lin_x = self._lerp_range(self.cfg.command_easy_lin_vel_x, self.cfg.command_hard_lin_vel_x, t)
        lin_y = self._lerp_range(self.cfg.command_easy_lin_vel_y, self.cfg.command_hard_lin_vel_y, t)
        yaw = self._lerp_range(self.cfg.command_easy_ang_vel_yaw, self.cfg.command_hard_ang_vel_yaw, t)
        heading = self._lerp_range(self.cfg.command_easy_heading, self.cfg.command_hard_heading, t)

        rand = torch.rand(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = rand * (lin_x[:, 1] - lin_x[:, 0]) + lin_x[:, 0]
        rand = torch.rand(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 1] = rand * (lin_y[:, 1] - lin_y[:, 0]) + lin_y[:, 0]
        if self.cfg.heading_command:
            rand = torch.rand(len(env_ids), device=self.device)
            self.heading_target[env_ids] = rand * (heading[:, 1] - heading[:, 0]) + heading[:, 0]
            self.is_heading_env[env_ids] = True
        else:
            rand = torch.rand(len(env_ids), device=self.device)
            self.vel_command_b[env_ids, 2] = rand * (yaw[:, 1] - yaw[:, 0]) + yaw[:, 0]
            self.is_heading_env[env_ids] = False

        self._apply_mirrored_commands(env_ids)
        standing_mask = self._apply_standing_commands(env_ids)
        self._apply_straight_commands(env_ids, standing_mask)

    def _apply_mirrored_commands(self, env_ids: Sequence[int]):
        env_ids = self._as_env_ids(env_ids)
        if not self.cfg.mirror_commands or len(env_ids) < 2:
            return
        pair_count = (len(env_ids) // 2) * 2
        left_ids = env_ids[:pair_count:2]
        right_ids = env_ids[1:pair_count:2]
        self.vel_command_b[right_ids, 0] = self.vel_command_b[left_ids, 0]
        self.vel_command_b[right_ids, 1] = -self.vel_command_b[left_ids, 1]
        if self.cfg.heading_command:
            self.heading_target[right_ids] = -self.heading_target[left_ids]
        else:
            self.vel_command_b[right_ids, 2] = -self.vel_command_b[left_ids, 2]

    def _apply_standing_commands(self, env_ids: Sequence[int]) -> torch.Tensor:
        env_ids = self._as_env_ids(env_ids)
        standing_mask = torch.rand(len(env_ids), device=self.device) < float(self.cfg.rel_standing_envs)
        if standing_mask.any():
            stand_env_ids = env_ids[standing_mask]
            self.vel_command_b[stand_env_ids, :3] = 0.0
            if self.cfg.heading_command:
                self.heading_target[stand_env_ids] = 0.0
        return standing_mask

    def _apply_straight_commands(self, env_ids: Sequence[int], standing_mask: torch.Tensor):
        env_ids = self._as_env_ids(env_ids)
        straight_prob = float(self.cfg.rel_straight_envs)
        if straight_prob <= 0.0 or len(env_ids) == 0:
            return

        candidate_mask = torch.logical_not(standing_mask)
        straight_mask = candidate_mask & (torch.rand(len(env_ids), device=self.device) < straight_prob)
        if not straight_mask.any():
            return

        straight_env_ids = env_ids[straight_mask]
        if self.cfg.straight_abs_lin_vel_x is None:
            vx_low, vx_high = self.cfg.straight_lin_vel_x
            rand = torch.rand(len(straight_env_ids), device=self.device)
            self.vel_command_b[straight_env_ids, 0] = rand * (vx_high - vx_low) + vx_low
        else:
            vx_low, vx_high = self.cfg.straight_abs_lin_vel_x
            rand = torch.rand(len(straight_env_ids), device=self.device)
            speed = rand * (vx_high - vx_low) + vx_low
            sign = torch.where(
                torch.rand(len(straight_env_ids), device=self.device) < float(self.cfg.straight_positive_prob),
                torch.ones(len(straight_env_ids), device=self.device),
                -torch.ones(len(straight_env_ids), device=self.device),
            )
            self.vel_command_b[straight_env_ids, 0] = speed * sign

        self.vel_command_b[straight_env_ids, 1] = 0.0
        self.vel_command_b[straight_env_ids, 2] = 0.0
        if self.cfg.heading_command:
            self.heading_target[straight_env_ids] = self.robot.data.heading_w[straight_env_ids]

    def _update_command(self):
        if self.cfg.heading_command:
            heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            self.vel_command_b[:, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

    def _as_env_ids(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class DeployVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`DeployVelocityCommand`."""

    class_type: type[CommandTerm] = DeployVelocityCommand

    curriculum_level_init: float = 0.1
    curriculum_level_min: float = 0.0
    curriculum_level_max: float = 1.0
    easy_mix_probability: float = 0.2
    easy_mix_max_level: float = 0.5

    command_easy_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    command_easy_lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    command_easy_ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)
    command_easy_heading: tuple[float, float] = (-3.14, 3.14)
    command_hard_lin_vel_x: tuple[float, float] = (-1.0, 2.0)
    command_hard_lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    command_hard_ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)
    command_hard_heading: tuple[float, float] = (-3.14, 3.14)

    mirror_commands: bool = False
    rel_straight_envs: float = 0.0
    straight_lin_vel_x: tuple[float, float] = (0.15, 1.20)
    straight_abs_lin_vel_x: tuple[float, float] | None = None
    straight_positive_prob: float = 0.5
