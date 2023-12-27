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
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


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
                                                nn.Conv2d(8, 8, kernel_size=(3, 2), stride=(1, 2)),
                                                nn.ReLU(),
                                                nn.Flatten()
                                                )
        self.mlp = nn.Sequential(nn.Linear(28, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, action_space)
                                         )
        # self.mean_layer = nn.Linear(64, self.num_actions)
        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        
    def compute(self, inputs, role):
        
        ee_states = inputs['states'][:, :12]
        d_imgs = inputs['states'][:, 12:].view(-1, 1, 48, 64)
        fetures = self.d_feture_extractor(d_imgs)
        combined = torch.cat([fetures, ee_states], dim=1)

        return self.mlp(combined), {}
    
    # def act(self, obs_buf, role):
    #     if role == "policy":
    #         return GaussianMixin.act(self, inputs, role)
    #     elif role == "value":
    #         return DeterministicMixin.act(self, inputs, role)
