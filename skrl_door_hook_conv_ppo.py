import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="DoorHook")
env = wrap_env(env)

device = env.device

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=2),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, stride=2),
                                                nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=2),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, stride=2),
                                                nn.Flatten()
                                                )
        self.mlp = nn.Sequential(nn.Linear(108, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 64),
                                        #  nn.ReLU(),
                                        #  nn.Linear(64, self.num_actions),
                                        #  nn.Tanh()
                                        )
                                         
        self.mean_layer = nn.Sequential(nn.Linear(64, self.num_actions),
                                        nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):

        if role == 'policy':
            return GaussianMixin.act(self, inputs , role)
        elif role == 'value':
            return DeterministicMixin.act(self, inputs, role)
        
    def compute(self, inputs, role):
        
        states = inputs['states']
        # print('states.shape', states.shape)
        ee_states = states[:, :12]
        # print(ee_states.shape)
        d_imgs = states[:, 12:]
        # print(d_imgs.shape)
        d_imgs = states[:, 12:].view(-1, 1, 48, 64)
        # print(d_imgs.shape)
        fetures = self.d_feture_extractor(d_imgs)
        # print(fetures.shape)
        combined = torch.cat([fetures, ee_states], dim=-1)
        # print(combined.shape)
        if role == 'policy':
            return self.mean_layer(self.mlp(combined)), self.log_std_parameter, {}
        elif role == 'value':
            return self.value_layer(self.mlp(combined)), {}
        




# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=128, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 128  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8  # 16 * 4096 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0.008
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 20
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "skrl_runs/DoorHook/conv_ppo"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# agent.load('skrl_runs/DoorHook/conv_ppo/23-12-28_01-42-12-425123_PPO/checkpoints/agent_1000.pt')


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 200000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
# trainer.train()
trainer.eval()



# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/IsaacGymEnvs-FrankaCabinet-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()
