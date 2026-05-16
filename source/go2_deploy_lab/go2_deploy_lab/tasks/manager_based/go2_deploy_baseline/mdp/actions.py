"""Action terms for the stateful Go2 deploy migration."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class DelayedNoisyJointPositionAction(JointPositionAction):
    """Joint-position action with gym deploy style delay and action noise."""

    cfg: "DelayedNoisyJointPositionActionCfg"

    def __init__(self, cfg: "DelayedNoisyJointPositionActionCfg", env):
        super().__init__(cfg, env)

        self._action_history = torch.zeros(
            self.num_envs,
            int(cfg.delay_max_steps) + 1,
            self.action_dim,
            device=self.device,
        )
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._applied_raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._last_applied_raw_actions = torch.zeros_like(self._applied_raw_actions)
        self._curriculum_level = float(cfg.curriculum_level_init)
        self._action_noise_std_cur = 0.0
        self._delay_max_cur = 0
        self.set_curriculum_level(self._curriculum_level)

    @property
    def applied_raw_actions(self) -> torch.Tensor:
        return self._applied_raw_actions

    @property
    def last_applied_raw_actions(self) -> torch.Tensor:
        return self._last_applied_raw_actions

    @property
    def delay_steps(self) -> torch.Tensor:
        return self._delay_steps

    @property
    def action_noise_std(self) -> float:
        return float(self._action_noise_std_cur)

    @property
    def delay_max_cur(self) -> int:
        return int(self._delay_max_cur)

    def set_curriculum_level(self, level: float):
        self._curriculum_level = float(
            min(max(level, float(self.cfg.curriculum_level_min)), float(self.cfg.curriculum_level_max))
        )
        self._action_noise_std_cur = self._curriculum_level * float(self.cfg.action_noise_std_max)
        delay_easy = int(self.cfg.delay_easy_max_steps)
        delay_hard = int(self.cfg.delay_max_steps)
        self._delay_max_cur = int(round(delay_easy + (delay_hard - delay_easy) * self._curriculum_level))

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
        self._action_history[:, 0, :] = actions

        env_ids = torch.arange(self.num_envs, device=self.device)
        delay_idx = self._delay_steps.clamp(min=0, max=self._action_history.shape[1] - 1)
        delayed_actions = self._action_history[env_ids, delay_idx]

        if self._action_noise_std_cur > 0.0:
            delayed_actions = delayed_actions + torch.randn_like(delayed_actions) * self._action_noise_std_cur

        self._applied_raw_actions[:] = torch.clamp(
            delayed_actions, -float(self.cfg.clip_actions), float(self.cfg.clip_actions)
        )
        self._processed_actions = self._applied_raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def commit_applied_actions(self):
        self._last_applied_raw_actions[:] = self._applied_raw_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._action_history[env_ids] = 0.0
        self._applied_raw_actions[env_ids] = 0.0
        self._last_applied_raw_actions[env_ids] = 0.0
        self._delay_steps[env_ids] = self._sample_delay_steps(env_ids)

    def _sample_delay_steps(self, env_ids) -> torch.Tensor:
        if isinstance(env_ids, slice):
            count = self.num_envs
        else:
            count = len(env_ids)
        if self._delay_max_cur <= 0:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        return torch.randint(0, self._delay_max_cur + 1, (count,), device=self.device, dtype=torch.long)


@configclass
class DelayedNoisyJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`DelayedNoisyJointPositionAction`."""

    class_type: type[ActionTerm] = DelayedNoisyJointPositionAction

    clip_actions: float = 100.0
    curriculum_level_init: float = 0.1
    curriculum_level_min: float = 0.0
    curriculum_level_max: float = 1.0
    action_noise_std_max: float = 0.1
    delay_easy_max_steps: int = 0
    delay_max_steps: int = 1
