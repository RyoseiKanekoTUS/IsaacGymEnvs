import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


class SACActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

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
                                         nn.Linear(64, self.num_actions),
                                         nn.Tanh()
                                        )
                                         
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        
    def compute(self, inputs, role):
        states = inputs['states']
        ee_states = states[:, :12]
        pp_d_imgs = states[:, 12:].view(-1, 1, 48, 64)
        fetures = self.d_feture_extractor(pp_d_imgs)
        return self.mlp(torch.cat([fetures, ee_states], dim=-1)), self.log_std_parameter, {}
    
class SACCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=2),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, stride=2),
                                                nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=2),
                                                nn.ReLU(),
                                                nn.MaxPool2d(2, stride=2),
                                                nn.Flatten()
                                                )
        self.mlp = nn.Sequential(nn.Linear(108 + self.num_actions, 128),
                                         nn.ReLU(),
                                         nn.Linear(128, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, 1)
                                        )
    def compute(self, inputs, role):
        states = inputs['states']
        ee_states = states[:, :12]
        pp_d_imgs = states[:, 12:].view(-1, 1, 48, 64)
        fetures = self.d_feture_extractor(pp_d_imgs)
        return self.mlp(torch.cat([fetures, ee_states, inputs['taken_actions']], dim=-1)), {}


class DoorHookTrainer(SACActor, SACCritic):
    def __init__(self):

        set_seed(42)
        
        self.env = load_isaacgym_env_preview4(task_name="DoorHook")
        self.env = wrap_env(self.env)
        self.device = self.env.device
        self.memory = RandomMemory(memory_size=256, num_envs=self.env.num_envs, device=self.device)

        self.models = {}
        self.models["policy"] = SACActor(self.env.observation_space, self.env.action_space, self.device, clip_actions=True)
        self.models["critic_1"] = SACCritic(self.env.observation_space, self.env.action_space, self.device)
        self.models["critic_2"] = SACCritic(self.env.observation_space, self.env.action_space, self.device)
        self.models["target_critic_1"] = SACCritic(self.env.observation_space, self.env.action_space, self.device)
        self.models["target_critic_2"] = SACCritic(self.env.observation_space, self.env.action_space, self.device)

        self.cfg = SAC_DEFAULT_CONFIG.copy()
        self.cfg["gradient_steps"] = 1
        self.cfg["batch_size"] = 2048
        self.cfg["discount_factor"] = 0.99
        self.cfg["polyak"] = 0.005
        self.cfg["actor_learning_rate"] = 5e-3
        self.cfg["critic_learning_rate"] = 5e-3
        self.cfg["random_timesteps"] = 80
        self.cfg["learning_starts"] = 80
        self.cfg["grad_norm_clip"] = 0
        self.cfg["learn_entropy"] = True
        self.cfg["entropy_learning_rate"] = 5e-3
        self.cfg["initial_entropy_value"] = 1.0
        self.cfg["state_preprocessor"] = RunningStandardScaler
        self.cfg["state_preprocessor_kwargs"] = {"size": self.env.observation_space, "device": self.device}
        # logging to TensorBoard and write checkpoints (in timesteps)
        
    def train(self, load_path=None):
        # logging to TensorBoard and write checkpoints (in timesteps)
        self.cfg["experiment"]["write_interval"] = 300
        self.cfg["experiment"]["checkpoint_interval"] = 2000
        self.cfg["experiment"]["directory"] = "skrl_runs/DoorHook/conv_sac"

        self.agent = SAC(models=self.models,
                        memory=self.memory,
                        cfg=self.cfg,
                        observation_space=self.env.observation_space,
                        action_space=self.env.action_space,
                        device=self.device)
        if load_path:
            self.agent.load(load_path)
        else:
            pass

        self.cfg_trainer = {"timesteps": 200000, "headless": False}
        self.trainer = SequentialTrainer(cfg=self.cfg_trainer, env=self.env, agents=self.agent)

        self.trainer.train()
    
    def eval(self, load_path=None):

        self.agent = SAC(models=self.models,
                memory=self.memory,
                cfg=self.cfg,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                device=self.device)

        if load_path:
            self.agent.load(load_path)
        else:
            pass

        self.cfg_trainer = {"timesteps": 10000, "headless": False}
        self.trainer = SequentialTrainer(cfg=self.cfg_trainer, env=self.env, agents=self.agent)

        self.trainer.eval()


if __name__ == '__main__':

    # path = 'skrl_runs/DoorHook/conv_sac/23-12-29_01-16-38-304979_SAC/checkpoints/agent_2000.pt'
    path = None
    DoorHookTrainer = DoorHookTrainer()
    DoorHookTrainer.eval(path)
    # DoorHookTrainer.train(path)



    
