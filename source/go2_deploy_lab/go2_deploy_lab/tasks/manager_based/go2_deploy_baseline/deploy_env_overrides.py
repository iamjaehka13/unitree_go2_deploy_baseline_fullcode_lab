from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from . import mdp as go2_mdp


POLICY_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def apply_deploy_baseline_overrides(env_cfg):
    """Match the training observation vector to the real Go2 deploy runner."""
    env_cfg.scene.num_envs = 4096
    env_cfg.scene.height_scanner = None

    # Real deploy observation order:
    # ang vel, gravity, command, joint pos, joint vel, last action, sin/cos phase.
    policy_obs = env_cfg.observations.policy
    policy_obs.base_lin_vel = None
    policy_obs.height_scan = None
    policy_obs.base_ang_vel.scale = 0.25
    policy_obs.velocity_commands.scale = (2.0, 2.0, 0.25)
    policy_obs.joint_pos.scale = 1.0
    policy_obs.joint_pos.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True)
    }
    policy_obs.joint_vel.scale = 0.05
    policy_obs.joint_vel.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True)
    }
    policy_obs.phase = ObsTerm(
        func=go2_mdp.gait_phase,
        params={
            "command_name": "base_velocity",
            "gait_period": 0.6,
            "command_threshold": 0.1,
            "stand_phase_lock": True,
        },
    )

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.debug_vis = False
    command_cfg.ranges.lin_vel_x = (-0.5, 1.0)
    command_cfg.ranges.lin_vel_y = (-0.5, 0.5)
    command_cfg.ranges.ang_vel_z = (-1.0, 1.0)
    command_cfg.ranges.heading = (0.0, 0.0)

    action_cfg = env_cfg.actions.joint_pos
    action_cfg.joint_names = POLICY_JOINT_NAMES
    action_cfg.preserve_order = True
