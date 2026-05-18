from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from .deploy_env_overrides import apply_deploy_baseline_overrides


@configclass
class Go2DeployFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_deploy_baseline_overrides(self)


@configclass
class Go2DeployFlatEnvCfg_PLAY(Go2DeployFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.actor.params["add_deploy_noise"] = False
        self.deploy_push_robots = False
        self.events.physics_material = None
        self.events.deploy_friction = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
