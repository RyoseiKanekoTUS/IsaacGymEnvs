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

import datetime
import csv
import os
import time
import shutil


class PPOnet(GaussianMixin, DeterministicMixin, Model):
    #     ############################################################################################
    # def __init__(self, observation_space, action_space, device, clip_actions=False,
    #              clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
    #     Model.__init__(self, observation_space, action_space, device)
    #     DeterministicMixin.__init__(self, clip_actions)
    #     GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    #     # # NW v4_4
    #     self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 8, kernel_size=9, stride=1, padding=1), # 8, 42, 58
    #                     nn.ReLU(),
    #                     nn.MaxPool2d(2, stride=2), #  8, 21, 29
    #                     nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=1), # 16, 17, 25
    #                     nn.ReLU(),
    #                     nn.MaxPool2d(2, stride=2), # 16, 8, 12
    #                     nn.Conv2d(16, 32, kernel_size=5, padding=1), # 32, 6, 10
    #                     nn.ReLU(),
    #                     # nn.MaxPool2d(2, stride=2, padding=1), # 32, 4, 6
    #                     nn.Flatten() # 1920
    #                     )
        
        
    #     self.mlp = nn.Sequential(nn.Linear((6+1920), 1024),
    #                 nn.ELU(),
    #                 nn.Linear(1024, 512),
    #                 nn.ELU(),
    #                 nn.Linear(512, 256),
    #                 nn.ELU(),
    #                 nn.Linear(256, 64),
    #                 nn.ELU()
    #                 )        
    #     self.mean_layer = nn.Sequential(nn.Linear(64, self.num_actions),
    #                                     nn.Tanh())
    #     self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    #     self.value_layer = nn.Linear(64, 1)

    ###############################################################################################
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)


        # # NW v4 + silhouette

        self.d_feture_extractor = nn.Sequential(nn.Conv2d(1, 8, kernel_size=9, stride=1, padding=1), # 8, 42, 58
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=1), #  8, 22, 30
                                nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=1), # 16, 18, 26
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, padding=1), # 16, 10, 14
                                nn.Conv2d(16, 32, kernel_size=5, padding=1), # 32, 8, 12
                                nn.ReLU(),
                                nn.Flatten(), # 3072
                                # nn.Sigmoid(),
                                )
        
        
        self.mlp = nn.Sequential(nn.Linear((30+3072), 1256),
                    nn.ELU(),
                    nn.Linear(1256, 512),
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
        
        # get state from inputs
        states = inputs['states']
        # get hand rot,pos states 0:27 <- hand_rot_state_vector, 27:30 <-norm_d_p_door_hand
        norm_hand_states = states[:, :30]
        # get d_img fetures 
        pp_d_imgs = states[:, 30:].view(-1, 1, 48, 64) # input to CNN
        d_fetures = self.d_feture_extractor(pp_d_imgs) # output from CNN
        # norm_d_fetures = d_fetures / (torch.norm(d_fetures, dim=1, keepdim=True) + 1e-8)

        # re-concat state vectors : input to mlp
        states_vectors = torch.cat([norm_hand_states, d_fetures], dim=-1)

        if role == 'policy':
            actions_from_mlp = self.mean_layer(self.mlp(states_vectors))
            return actions_from_mlp, self.log_std_parameter, {}
        elif role == 'value':
            return self.value_layer(self.mlp(states_vectors)), {}


class DoorHookTrainer(PPOnet):
    def __init__(self):

        # set_seed(210)
        
        self.env = load_isaacgym_env_preview4(task_name="DoorHook")
        self.env = wrap_env(self.env)
        self.device = self.env.device
        self.memory = RandomMemory(memory_size=self.env.max_episode_length, num_envs=self.env.num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = PPOnet(self.env.observation_space, self.env.action_space, self.device)
        self.models["value"] = self.models["policy"]  # same instance: shared model

        self.cfg = PPO_DEFAULT_CONFIG.copy()
        self.cfg["rollouts"] = self.memory.memory_size # = self.env.max_episode_length
        self.cfg["learning_epochs"] = 12
        self.cfg["mini_batches"] = 60  # mem_size*num_envs / mini_batches : included in each mini_batch
        self.cfg["discount_factor"] = 0.99
        self.cfg["lambda"] = 0.95
        self.cfg["learning_rate"] = 1e-3
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
        self.cfg["state_preprocessor"] = None
        self.cfg["state_preprocessor_kwargs"] = {"size": self.env.observation_space, "device": self.device}
        self.cfg["value_preprocessor"] = None
        self.cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}
        # # logging to TensorBoard and write checkpoints (in timesteps)
        # self.cfg["experiment"]["write_interval"] = 20
        # self.cfg["experiment"]["checkpoint_interval"] = 1000
        # self.cfg["experiment"]["directory"] = "skrl_runs/DoorHook/conv_ppo"

        
    def train(self, load_path=None):
        # logging to TensorBoard and write checkpoints (in timesteps)
        self.cfg["experiment"]["write_interval"] = 20
        self.cfg["experiment"]["checkpoint_interval"] = 1000
        self.cfg["experiment"]["directory"] = "skrl_runs/DoorHook/sim2real_door_tf_action"

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

        door_hook_py_path = './isaacgymenvs/tasks/door_hook.py'
        dest_dir = './skrl_runs/DoorHook/cfg'
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        cfg_file_name = f'door_hook_{self.env.begin_datetime}.py'
        dest_file = os.path.join(dest_dir, cfg_file_name)

        shutil.copy2(door_hook_py_path, dest_file)


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
        print("start")
        self.trainer.eval()
        
        print(self.trainer.env.statistics)
        flattened_data = []

        for sublist in self.env.statistics:
            for item in sublist:
                flattened_data.append(item)
        
        filename = f"output_{self.trainer.env.begin_datetime}.csv"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'statistic_data')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(flattened_data)

        print(f"writing done: {file_path}")


if __name__ == '__main__':

    path = None
    # path = '../../learning_data/DoorHook_POST_GRADUATION/non_vel_new_CNN_0319/best_agent.pt'
    # path = '../../learning_data/DoorHook_POST_GRADUATION/UR3_non_vel_405/0305_opening/best_agent.pt'
    # path = 'skrl_runs/DoorHook/non_vel/24-03-19_13-04-08-617586_PPO/checkpoints/best_agent.pt'
    
    DoorHookTrainer = DoorHookTrainer()
    DoorHookTrainer.eval(path)
    # DoorHookTrainer.train(path)



    
