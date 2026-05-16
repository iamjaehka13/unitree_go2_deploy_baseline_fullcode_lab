import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Deploy-Baseline-v0",
    entry_point=f"{__name__}.deploy_env:Go2DeployManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2DeployFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2DeployFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Deploy-Baseline-Play-v0",
    entry_point=f"{__name__}.deploy_env:Go2DeployManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2DeployFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2DeployFlatPPORunnerCfg",
    },
)
