"""Event terms for the stateful Go2 deploy migration."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.utils import math as math_utils


class randomize_deploy_friction(ManagerTermBase):
    """Apply gym deploy per-env global friction randomization."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_range: tuple[float, float],
        restitution: float = 0.0,
        num_buckets: int = 64,
        global_dr_update_interval_resets: int = 200,
        force: bool = False,
        use_buckets: bool = False,
    ):
        _ensure_deploy_friction_state(env, global_dr_update_interval_resets)
        if env_ids is None:
            env_ids_device = torch.arange(env.scene.num_envs, device=env.device)
        else:
            env_ids_device = env_ids.to(device=env.device)

        if force:
            update_env_ids = env_ids_device
        else:
            env._deploy_global_dr_counter[env_ids_device] += 1
            update_mask = torch.logical_or(
                torch.logical_not(env._deploy_global_dr_initialized[env_ids_device]),
                env._deploy_global_dr_counter[env_ids_device] >= int(global_dr_update_interval_resets),
            )
            update_env_ids = env_ids_device[update_mask]

        if len(update_env_ids) == 0:
            return

        if use_buckets:
            bucket_ids = torch.randint(0, int(num_buckets), (len(update_env_ids), 1), device=env.device)
            buckets = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (int(num_buckets), 1), device=env.device
            )
            coeffs = buckets[bucket_ids.squeeze(-1)]
        else:
            coeffs = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (len(update_env_ids), 1), device=env.device
            )

        env._deploy_friction_coeffs_cur[update_env_ids] = coeffs

        if not force:
            env._deploy_global_dr_counter[update_env_ids] = 0
            env._deploy_global_dr_initialized[update_env_ids] = True

        materials = self.asset.root_physx_view.get_material_properties()
        env_ids_cpu = update_env_ids.detach().cpu()
        coeffs_cpu = coeffs.detach().cpu()
        samples = torch.zeros((len(update_env_ids), self.asset.root_physx_view.max_shapes, 3), device="cpu")
        coeffs_cpu = coeffs_cpu.expand(-1, self.asset.root_physx_view.max_shapes)
        samples[:, :, 0] = coeffs_cpu
        samples[:, :, 1] = coeffs_cpu
        samples[:, :, 2] = float(restitution)
        materials[env_ids_cpu] = samples
        self.asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def push_by_setting_velocity_with_history(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set root xy velocity like gym deploy and record privileged push history."""
    if len(env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    root_vel = asset.data.root_vel_w[env_ids].clone()
    ranges = torch.tensor(
        [velocity_range.get("x", (0.0, 0.0)), velocity_range.get("y", (0.0, 0.0))],
        device=asset.device,
    )
    sampled_xy = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=asset.device)
    root_vel[:, 0:2] = sampled_xy
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    env._deploy_push_history_xy[env_ids] = sampled_xy


def _ensure_deploy_friction_state(env, global_dr_update_interval_resets: int):
    if hasattr(env, "_deploy_friction_coeffs_cur"):
        return
    env._deploy_friction_coeffs_cur = torch.ones(env.num_envs, 1, device=env.device)
    env._deploy_global_dr_counter = torch.full(
        (env.num_envs,),
        int(global_dr_update_interval_resets),
        dtype=torch.long,
        device=env.device,
    )
    env._deploy_global_dr_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
