import isaacgym
from skrl.envs.wrappers.torch import wrap_env
import isaacgymenvs

env = isaacgymenvs.make(seed=0,
                        task="DoorHook",
                        num_envs=512,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=False)

env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'
