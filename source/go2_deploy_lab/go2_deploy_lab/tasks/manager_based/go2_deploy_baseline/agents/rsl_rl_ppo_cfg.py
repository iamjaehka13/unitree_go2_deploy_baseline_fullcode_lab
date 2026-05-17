from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class Go2DeployFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.seed = 1
        self.clip_actions = 100.0
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.max_iterations = 5000
        self.save_interval = 50
        self.experiment_name = "flat_go2_deploy"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
