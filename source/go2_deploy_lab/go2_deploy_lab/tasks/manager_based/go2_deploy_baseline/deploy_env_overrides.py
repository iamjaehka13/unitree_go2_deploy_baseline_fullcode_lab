import math

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

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

FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class DeployPolicyObsCfg(ObsGroup):
    """Single-term 47D actor observation with gym deploy noise timing."""

    actor = ObsTerm(
        func=go2_mdp.deploy_actor_observation,
        clip=(-100.0, 100.0),
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "gait_period": 0.6,
            "command_threshold": 0.1,
            "stand_phase_lock": True,
            "add_noise": True,
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DeployCriticObsCfg(ObsGroup):
    """80D privileged critic observation matching gym deploy."""

    actor = ObsTerm(
        func=go2_mdp.deploy_actor_observation,
        clip=(-100.0, 100.0),
        params={
            "command_name": "base_velocity",
            "action_name": "joint_pos",
            "gait_period": 0.6,
            "command_threshold": 0.1,
            "stand_phase_lock": True,
            "add_noise": False,
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True),
        },
    )
    base_lin_vel = ObsTerm(func=go2_mdp.base_lin_vel_scaled, clip=(-100.0, 100.0))
    feet_pos_z = ObsTerm(
        func=go2_mdp.feet_pos_z,
        clip=(-100.0, 100.0),
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True)},
    )
    feet_air_time = ObsTerm(
        func=go2_mdp.feet_air_time,
        clip=(-100.0, 100.0),
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True)},
    )
    foot_contact = ObsTerm(
        func=go2_mdp.foot_contact,
        clip=(-100.0, 100.0),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "threshold": 1.0,
        },
    )
    contact_forces = ObsTerm(
        func=go2_mdp.contact_forces,
        clip=(-100.0, 100.0),
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True)},
    )
    friction_coeffs = ObsTerm(func=go2_mdp.deploy_friction_coeffs, clip=(-100.0, 100.0))
    delay_steps = ObsTerm(
        func=go2_mdp.deploy_delay_steps,
        clip=(-100.0, 100.0),
        params={"action_name": "joint_pos"},
    )
    action_noise_std = ObsTerm(
        func=go2_mdp.deploy_action_noise_std,
        clip=(-100.0, 100.0),
        params={"action_name": "joint_pos"},
    )
    obs_noise_level = ObsTerm(func=go2_mdp.deploy_obs_noise_level, clip=(-100.0, 100.0))
    push_history_xy = ObsTerm(func=go2_mdp.deploy_push_history_xy, clip=(-100.0, 100.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def apply_deploy_baseline_overrides(env_cfg):
    """Apply the cleaned gym Go2 deploy training semantics to the Lab task."""
    _apply_scalar_deploy_cfg(env_cfg)
    _apply_scene_and_robot_cfg(env_cfg)
    _apply_observation_cfg(env_cfg)
    _apply_command_cfg(env_cfg)
    _apply_action_cfg(env_cfg)
    _apply_event_cfg(env_cfg)
    _apply_reward_cfg(env_cfg)
    _apply_termination_cfg(env_cfg)
    _apply_curriculum_cfg(env_cfg)


def _apply_scalar_deploy_cfg(env_cfg):
    env_cfg.seed = 1
    env_cfg.deploy_curriculum_level_init = 0.1
    env_cfg.deploy_curriculum_level_min = 0.0
    env_cfg.deploy_curriculum_level_max = 1.0
    env_cfg.deploy_curriculum_step_up = 0.01
    env_cfg.deploy_curriculum_step_down = 0.03
    env_cfg.deploy_curriculum_ema_alpha = 0.03
    env_cfg.deploy_curriculum_ready_timeout_rate = 0.80
    env_cfg.deploy_curriculum_ready_tracking = 0.75
    env_cfg.deploy_curriculum_ready_fall_rate = 0.15
    env_cfg.deploy_curriculum_ready_streak = 4
    env_cfg.deploy_curriculum_hard_fall_rate = 0.25
    env_cfg.deploy_curriculum_hard_streak = 2
    env_cfg.deploy_curriculum_cooldown = 5
    env_cfg.deploy_obs_noise_level_max = 1.0
    env_cfg.deploy_action_noise_std_max = 0.1
    env_cfg.deploy_delay_easy_max_steps = 0
    env_cfg.deploy_delay_max_steps = 1
    env_cfg.deploy_global_dr_update_interval_resets = 200
    env_cfg.deploy_push_robots = True
    env_cfg.deploy_push_interval_s = 5.0
    env_cfg.deploy_max_push_vel_xy = 0.5
    env_cfg.deploy_tracking_lin_vel_scale = 3.5
    env_cfg.deploy_tracking_ang_vel_scale = 1.75


def _apply_scene_and_robot_cfg(env_cfg):
    env_cfg.scene.num_envs = 4096
    env_cfg.scene.env_spacing = 3.0
    env_cfg.scene.height_scanner = None

    robot_cfg = env_cfg.scene.robot
    robot_cfg.init_state.pos = (0.0, 0.0, 0.42)
    actuator_cfg = robot_cfg.actuators["base_legs"]
    actuator_cfg.stiffness = 20.0
    actuator_cfg.damping = 0.5
    actuator_cfg.effort_limit = {
        ".*_hip_joint": 23.7,
        ".*_thigh_joint": 23.7,
        ".*_calf_joint": 35.55,
    }
    actuator_cfg.velocity_limit = {
        ".*_hip_joint": 30.1,
        ".*_thigh_joint": 30.1,
        ".*_calf_joint": 20.07,
    }
    actuator_cfg.saturation_effort = 35.55


def _apply_observation_cfg(env_cfg):
    env_cfg.observations.policy = DeployPolicyObsCfg()
    env_cfg.observations.critic = DeployCriticObsCfg()


def _apply_command_cfg(env_cfg):
    # The curriculum's hard range is intentionally wider than the real deploy
    # runner's conservative default command limits.
    env_cfg.commands.base_velocity = go2_mdp.DeployVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        debug_vis=False,
        ranges=go2_mdp.DeployVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
        curriculum_level_init=0.1,
        curriculum_level_min=0.0,
        curriculum_level_max=1.0,
        easy_mix_probability=0.2,
        easy_mix_max_level=0.5,
        command_easy_lin_vel_x=(-0.5, 1.0),
        command_easy_lin_vel_y=(-0.5, 0.5),
        command_easy_ang_vel_yaw=(-1.0, 1.0),
        command_easy_heading=(-3.14, 3.14),
        command_hard_lin_vel_x=(-1.0, 2.0),
        command_hard_lin_vel_y=(-1.0, 1.0),
        command_hard_ang_vel_yaw=(-1.0, 1.0),
        command_hard_heading=(-3.14, 3.14),
        mirror_commands=False,
        rel_straight_envs=0.0,
        straight_lin_vel_x=(0.15, 1.20),
        straight_abs_lin_vel_x=None,
        straight_positive_prob=0.5,
    )


def _apply_action_cfg(env_cfg):
    env_cfg.actions.joint_pos = go2_mdp.DelayedNoisyJointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip_actions=100.0,
        curriculum_level_init=0.1,
        curriculum_level_min=0.0,
        curriculum_level_max=1.0,
        action_noise_std_max=0.1,
        delay_easy_max_steps=0,
        delay_max_steps=1,
    )


def _apply_event_cfg(env_cfg):
    friction_params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "friction_range": (0.3, 1.25),
        "restitution": 0.0,
        "num_buckets": 64,
        "global_dr_update_interval_resets": 200,
    }
    env_cfg.events.physics_material = EventTerm(
        func=go2_mdp.randomize_deploy_friction,
        mode="startup",
        params={**friction_params, "force": True, "use_buckets": True},
    )
    env_cfg.events.deploy_friction = EventTerm(
        func=go2_mdp.randomize_deploy_friction,
        mode="reset",
        params={**friction_params, "force": False, "use_buckets": False},
    )
    # Keep the Lab baseline inside its training distribution: no base-mass DR.
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        },
    }


def _apply_reward_cfg(env_cfg):
    for reward_name in list(env_cfg.rewards.__dict__.keys()):
        if not reward_name.startswith("_"):
            delattr(env_cfg.rewards, reward_name)

    joint_asset = SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True)
    foot_sensor = SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True)
    foot_asset = SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True)

    env_cfg.rewards.action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    env_cfg.rewards.air_time_variance = RewTerm(
        func=go2_mdp.air_time_variance,
        weight=-0.5,
        params={"sensor_cfg": foot_sensor},
    )
    env_cfg.rewards.ang_vel_xy = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)
    env_cfg.rewards.collision = RewTerm(
        func=go2_mdp.undesired_contacts_current,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
            "threshold": 0.1,
        },
    )
    env_cfg.rewards.dof_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": joint_asset})
    env_cfg.rewards.dof_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": joint_asset},
    )
    env_cfg.rewards.dof_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": joint_asset})
    env_cfg.rewards.feet_air_time = RewTerm(
        func=go2_mdp.feet_air_time_deploy,
        weight=3.0,
        params={"sensor_cfg": foot_sensor, "command_name": "base_velocity", "threshold": 0.5},
    )
    env_cfg.rewards.feet_slide = RewTerm(
        func=go2_mdp.feet_slide,
        weight=-0.5,
        params={"sensor_cfg": foot_sensor, "asset_cfg": foot_asset, "threshold": 1.0},
    )
    env_cfg.rewards.foot_gait = RewTerm(
        func=go2_mdp.foot_gait,
        weight=0.10,
        params={
            "command_name": "base_velocity",
            "gait_period": 0.6,
            "gait_stance_ratio": 0.56,
            "gait_offsets": (0.0, 0.5, 0.5, 0.0),
            "command_threshold": 0.1,
            "sensor_cfg": foot_sensor,
        },
    )
    env_cfg.rewards.joint_pos = RewTerm(
        func=go2_mdp.joint_pos_deploy,
        weight=-0.3,
        params={
            "command_name": "base_velocity",
            "body_speed_threshold": 0.3,
            "stand_still_scale": 5.0,
            "asset_cfg": joint_asset,
        },
    )
    env_cfg.rewards.lin_vel_z = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-2.0)
    env_cfg.rewards.orientation = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.5)
    env_cfg.rewards.torques = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=-0.0002,
        params={"asset_cfg": joint_asset},
    )
    env_cfg.rewards.tracking_ang_vel = RewTerm(
        func=base_mdp.track_ang_vel_z_exp,
        weight=1.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    env_cfg.rewards.tracking_lin_vel = RewTerm(
        func=base_mdp.track_lin_vel_xy_exp,
        weight=3.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    env_cfg.rewards.termination = RewTerm(func=go2_mdp.termination, weight=-200.0)


def _apply_termination_cfg(env_cfg):
    contact_bodies = ["base", ".*_hip", ".*_thigh", ".*_calf"]
    env_cfg.terminations.time_out = DoneTerm(func=go2_mdp.time_out_after_max_length, time_out=True)
    env_cfg.terminations.base_contact = DoneTerm(
        func=go2_mdp.illegal_contact_current,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=contact_bodies),
            "threshold": 1.0,
        },
    )
    env_cfg.terminations.bad_roll_pitch = DoneTerm(
        func=go2_mdp.bad_roll_pitch,
        params={"roll_limit": 0.8, "pitch_limit": 1.0},
    )


def _apply_curriculum_cfg(env_cfg):
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.deploy = CurrTerm(
        func=go2_mdp.deploy_command_curriculum,
        params={
            "tracking_lin_reward_name": "tracking_lin_vel",
            "tracking_ang_reward_name": "tracking_ang_vel",
            "command_name": "base_velocity",
            "action_name": "joint_pos",
        },
    )
