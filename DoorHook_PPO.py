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


class PPOnet(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # # NW v4_4
        self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 8, kernel_size=9, stride=1, padding=1), # 8, 42, 58
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2), #  8, 21, 29
                        nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=1), # 16, 17, 25
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2), # 16, 8, 12
                        nn.Conv2d(16, 32, kernel_size=5, padding=1), # 32, 6, 10
                        nn.ReLU(),
                        # nn.MaxPool2d(2, stride=2, padding=1), # 32, 4, 6
                        nn.Flatten() # 1920
                        )
        
        
        self.mlp = nn.Sequential(nn.Linear((6+1920), 1024),
                    nn.ELU(),
                    nn.Linear(1024, 512),
                    nn.ELU(),
                    nn.Linear(512, 256),
                    nn.ELU(),
                    nn.Linear(256, 64),
                    nn.ELU()
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
        # print('$ states shape $$$$$$$$$$$$$$$$$$$$$$$$$$$$$',states.shape)
        ee_states = states[:, :6]

        pp_d_imgs = states[:, 6:].view(-1, 1, 48, 64)
        d_feture = self.d_feture_extractor(pp_d_imgs)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$", d_feture.shape)

        # dist_d_imgs = states[:, 3084:].view(-1, 1, 48, 64)
        # distance = self.depth_extractor(dist_d_imgs)
        combined = torch.cat([ee_states, d_feture], dim=-1)
        if role == 'policy':
            return self.mean_layer(self.mlp(combined)), self.log_std_parameter, {}
        elif role == 'value':
            return self.value_layer(self.mlp(combined)), {}


class DoorHookTrainer(PPOnet):
    def __init__(self):

        set_seed(210)
        
        self.env = load_isaacgym_env_preview4(task_name="DoorHook")
        self.env = wrap_env(self.env)
        self.device = self.env.device
        self.memory = RandomMemory(memory_size=300, num_envs=self.env.num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = PPOnet(self.env.observation_space, self.env.action_space, self.device)
        self.models["value"] = self.models["policy"]  # same instance: shared model

        self.cfg = PPO_DEFAULT_CONFIG.copy()
        self.cfg["rollouts"] = 300  # memory_size
        self.cfg["learning_epochs"] = 24
        self.cfg["mini_batches"] = 128  # 16 * 4096 / 8192
        self.cfg["discount_factor"] = 0.99
        self.cfg["lambda"] = 0.95
        self.cfg["learning_rate"] = 5e-3
        self.cfg["learning_rate_scheduler"] = KLAdaptiveRL
        self.cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        self.cfg["random_timesteps"] = 0
        self.cfg["learning_starts"] = 0
        self.cfg["grad_norm_clip"] = 1.0
        self.cfg["ratio_clip"] = 0.2
        self.cfg["value_clip"] = 0.2
        self.cfg["clip_predicted_values"] = False
        self.cfg["entropy_loss_scale"] = 0.0
        self.cfg["value_loss_scale"] = 2.0
        self.cfg["kl_threshold"] = 0
        self.cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
        self.cfg["state_preprocessor"] = RunningStandardScaler
        self.cfg["state_preprocessor_kwargs"] = {"size": self.env.observation_space, "device": self.device}
        self.cfg["value_preprocessor"] = RunningStandardScaler
        self.cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}
        # # logging to TensorBoard and write checkpoints (in timesteps)
        # self.cfg["experiment"]["write_interval"] = 20
        # self.cfg["experiment"]["checkpoint_interval"] = 1000
        # self.cfg["experiment"]["directory"] = "skrl_runs/DoorHook/conv_ppo"

        
    def train(self, load_path=None):
        # logging to TensorBoard and write checkpoints (in timesteps)
        self.cfg["experiment"]["write_interval"] = 20
        self.cfg["experiment"]["checkpoint_interval"] = 1000
        self.cfg["experiment"]["directory"] = "skrl_runs/DoorHook/non_vel"

        self.agent = PPO(models=self.models,
                        memory=self.memory,
                        cfg=self.cfg,
                        observation_space=self.env.observation_space,
                        action_space=self.env.action_space,
                        device=self.device)
        if load_path:
            self.agent.load(load_path)
        else:
            pass

        self.cfg_trainer = {"timesteps": 250000, "headless": False}
        self.trainer = SequentialTrainer(cfg=self.cfg_trainer, env=self.env, agents=self.agent)

        self.trainer.train()
    
    def eval(self, load_path=None):

        self.agent = PPO(models=self.models,
                memory=self.memory,
                cfg=self.cfg,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                device=self.device)

        if load_path:
            self.agent.load(load_path)
        else:
            pass

        self.cfg_trainer = {"timesteps": 120000, "headless": False}
        self.trainer = SequentialTrainer(cfg=self.cfg_trainer, env=self.env, agents=self.agent)

        self.trainer.eval()


if __name__ == '__main__':

    path = None
    path = '../../learning_data/DoorHook_POST_GRADUATION/non_vel_new_CNN_0319/agent_250000.pt'
    # path = 'skrl_runs/DoorHook/non_vel/24-03-19_13-04-08-617586_PPO/checkpoints/best_agent.pt'
    
    DoorHookTrainer = DoorHookTrainer()
    DoorHookTrainer.eval(path)
    # DoorHookTrainer.train(path)



    
