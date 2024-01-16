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

        # # NW_v1
        # self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=2),
        #                                         nn.ELU(),
        #                                         nn.MaxPool2d(2, stride=2),
        #                                         nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=2),
        #                                         nn.ELU(),
        #                                         nn.MaxPool2d(2, stride=2),
        #                                         nn.Flatten()
        #                                         )
        # self.mlp = nn.Sequential(nn.Linear(108, 256),
        #                                  nn.ELU(),
        #                                  nn.Linear(256, 64),
        #                                  nn.ELU()
        #                                 )

        # NW_v2_Max pull push both
        # self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), # (2,48,64) 6144
        #                                         nn.ELU(),
        #                                         nn.MaxPool2d(2, stride=2), # (4,24,32) 3072
        #                                         nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # (8,24,32) 6144
        #                                         nn.ELU(),
        #                                         nn.MaxPool2d(2, stride=2), # (8,12,16) 1536
        #                                         nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (16,6,8) 768
        #                                         # nn.ELU(), # comment out 
        #                                         # nn.MaxPool2d(2, stride=1), # (16,5,7) 560 # comment out
        #                                         nn.Flatten()
        #                                         )
        # self.mlp = nn.Sequential(nn.Linear((12+768), 512),  # 560/768
        #                          nn.ELU(),
        #                          nn.Linear(512, 256),
        #                          nn.ELU(),
        #                          nn.Linear(256, 64),
        #                          nn.ELU()
        #                          )        
        
        # NW_v2_Avg
        # self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), # (2,48,64) 6144
        #                                         nn.ELU(),
        #                                         nn.AvgPool2d(2, stride=2), # (4,24,32) 3072
        #                                         nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # (8,24,32) 6144
        #                                         nn.ELU(),
        #                                         nn.AvgPool2d(2, stride=2), # (8,12,16) 1536
        #                                         nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (16,6,8) 768
        #                                         nn.ELU(),
        #                                         nn.AvgPool2d(2, stride=1), # (16,5,7) 560
        #                                         nn.Flatten()
        #                                         )
        # self.mlp = nn.Sequential(nn.Linear((12+560), 512),
        #                          nn.ELU(),
        #                          nn.Linear(512, 256),
        #                          nn.ELU(),
        #                          nn.Linear(256, 64),
        #                          nn.ELU()
        #                          )        
        
        # # NW_v2.2_Max
        # self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), # (2,48,64) 6144
        #                              nn.ELU(),
        #                              nn.MaxPool2d(4, stride=2, padding=1), # (4,24,32) 3072
        #                              nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # (8,24,32) 6144
        #                              nn.ELU(),
        #                              nn.MaxPool2d(4, stride=2, padding=1), # (8,12,16) 1536
        #                              nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1), # (24,12,16) 4608
        #                              nn.ELU(),
        #                              nn.MaxPool2d(4, stride=2), # (24,5,7) # 840
        #                              nn.Conv2d(24, 24, kernel_size=4, stride=1, padding=2), # (24, 6, 8) 1152
        #                              nn.ELU(),
        #                              nn.MaxPool2d(2, stride=2), # (24, 3, 4) # 288
        #                              nn.Flatten()
        #                              )
        
        # self.mlp = nn.Sequential(nn.Linear((12+288), 512),
        #                     nn.ELU(),
        #                     nn.Linear(512, 256),
        #                     nn.ELU(),
        #                     nn.Linear(256, 64),
        #                     nn.ELU()
        #                     )        

        # # NW v3_Max
        # self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), # (2,48,64) 6144
        #                         nn.ELU(),
        #                         nn.MaxPool2d(4, stride=2, padding=1), # (4,24,32) 3072
        #                         nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # (8,24,32) 6144
        #                         nn.ELU(),
        #                         nn.MaxPool2d(4, stride=2, padding=1), # (8,12,16) 1536
        #                         nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1), # (24,12,16) 4608
        #                         nn.ELU(),
        #                         nn.MaxPool2d(4, stride=2), # (24,5,7) # 840
        #                         nn.Conv2d(24, 24, kernel_size=4, stride=1, padding=2), # (24, 6, 8) 1152
        #                         nn.ELU(),
        #                         nn.Flatten()
        #                         )
        # self.mlp = nn.Sequential(nn.Linear((12+1152), 512),
        #             nn.ELU(),
        #             nn.Linear(512, 256),
        #             nn.ELU(),
        #             nn.Linear(256, 64),
        #             nn.ELU()
        #             )        
        # self.mean_layer = nn.Sequential(nn.Linear(64, self.num_actions),
        #                                 nn.Tanh())
        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # self.value_layer = nn.Linear(64, 1)

        # # NW v4 + silhouette
        self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 4, kernel_size=9, stride=1, padding=1), # 8, 42, 58
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=1), #  8, 22, 30
                                nn.Conv2d(4, 8, kernel_size=7, stride=1, padding=1), # 16, 18, 26
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=1), # 16, 10, 14
                                nn.Conv2d(8, 16, kernel_size=5, padding=1), # 32, 8, 12
                                nn.ReLU(),
                                nn.Flatten()
                                )
        
        # self.depth_extractor = self.silhouette_extractor
        
        self.mlp = nn.Sequential(nn.Linear((12+1536), 512),
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
        ee_states = states[:, :12]

        pp_d_imgs = states[:, 12:].view(-1, 1, 48, 64)
        d_feture = self.d_feture_extractor(pp_d_imgs)

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
        self.memory = RandomMemory(memory_size=256, num_envs=self.env.num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = PPOnet(self.env.observation_space, self.env.action_space, self.device)
        self.models["value"] = self.models["policy"]  # same instance: shared model

        self.cfg = PPO_DEFAULT_CONFIG.copy()
        self.cfg["rollouts"] = 256  # memory_size
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
        self.cfg["experiment"]["directory"] = "skrl_runs/DoorHook/conv_ppo"

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

        self.cfg_trainer = {"timesteps": 500000, "headless": False}
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

        self.cfg_trainer = {"timesteps": 1200, "headless": False}
        self.trainer = SequentialTrainer(cfg=self.cfg_trainer, env=self.env, agents=self.agent)

        self.trainer.eval()


if __name__ == '__main__':

    path = None
    # path = '../../learning_data/DoorHook/skrl/0111_uni_pull_push_both/best_agent.pt'
    path = 'skrl_runs/DoorHook/conv_ppo/24-01-16_17-45-53-591374_PPO_hand_up_some_open/checkpoints/best_agent.pt'
    
    DoorHookTrainer = DoorHookTrainer()
    DoorHookTrainer.eval(path)
    # DoorHookTrainer.train(path)



    
