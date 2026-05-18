"""Stateful manager-based environment for the Go2 deploy migration."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from . import mdp as go2_mdp


class PositiveClipTerminationRewardManager(RewardManager):
    """Reward manager matching gym: positive clip first, terminal penalty after."""

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        termination_term = None

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if name == "termination":
                termination_term = (term_idx, name, term_cfg)
                self._step_reward[:, term_idx] = 0.0
                continue
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

        self._reward_buf[:] = torch.clamp(self._reward_buf, min=0.0)

        if termination_term is not None:
            term_idx, name, term_cfg = termination_term
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf


class Go2DeployManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with gym deploy step ordering and stateful deploy buffers."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._init_deploy_state(cfg)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._sync_deploy_terms()

    def _init_deploy_state(self, cfg):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self._deploy_curriculum_level = float(cfg.deploy_curriculum_level_init)
        self._deploy_tracking_ema = 0.0
        self._deploy_timeout_rate_ema = 0.0
        self._deploy_fall_rate_ema = 0.0
        self._deploy_ready_streak = 0
        self._deploy_hard_streak = 0
        self._deploy_curriculum_cooldown = 0
        self.reset_time_outs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._deploy_obs_noise_level_cur = self._deploy_curriculum_level * float(cfg.deploy_obs_noise_level_max)
        self._deploy_action_noise_std_cur = self._deploy_curriculum_level * float(cfg.deploy_action_noise_std_max)
        delay_easy = int(cfg.deploy_delay_easy_max_steps)
        delay_hard = int(cfg.deploy_delay_max_steps)
        self._deploy_delay_max_cur = int(round(delay_easy + (delay_hard - delay_easy) * self._deploy_curriculum_level))
        self._deploy_phase = torch.zeros(num_envs, device=device)
        self._deploy_push_history_xy = torch.zeros(num_envs, 2, device=device)
        self._deploy_feet_air_time = torch.zeros(num_envs, 4, device=device)
        self._deploy_last_contacts = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self._deploy_friction_coeffs_cur = torch.ones(num_envs, 1, device=device)
        self._deploy_global_dr_counter = torch.full(
            (num_envs,),
            int(cfg.deploy_global_dr_update_interval_resets),
            dtype=torch.long,
            device=device,
        )
        self._deploy_global_dr_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _sync_deploy_terms(self):
        if not hasattr(self, "command_manager") or not hasattr(self, "action_manager"):
            return
        self.command_manager.get_term("base_velocity").set_curriculum_level(self._deploy_curriculum_level)
        self.action_manager.get_term("joint_pos").set_curriculum_level(self._deploy_curriculum_level)

    def load_managers(self):
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        ManagerBasedEnv.load_managers(self)

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_manager = PositiveClipTerminationRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._deploy_phase[:] = torch.remainder(self.episode_length_buf.float() * self.step_dt, 0.6) / 0.6
        self._deploy_push_history_xy.zero_()

        self.command_manager.compute(dt=self.step_dt)

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self._apply_deploy_pushes()

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self._commit_applied_actions()
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _apply_deploy_pushes(self):
        if not getattr(self.cfg, "deploy_push_robots", True):
            return
        interval = max(1, int(round(float(self.cfg.deploy_push_interval_s) / self.step_dt)))
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % interval == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = float(self.cfg.deploy_max_push_vel_xy)
        go2_mdp.push_by_setting_velocity_with_history(
            self,
            push_env_ids,
            velocity_range={"x": (-max_vel, max_vel), "y": (-max_vel, max_vel)},
        )

    def _commit_applied_actions(self):
        action_term = self.action_manager.get_term("joint_pos")
        if hasattr(action_term, "commit_applied_actions"):
            action_term.commit_applied_actions()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._deploy_feet_air_time[env_ids] = 0.0
        self._deploy_last_contacts[env_ids] = False
        self._sync_deploy_terms()
