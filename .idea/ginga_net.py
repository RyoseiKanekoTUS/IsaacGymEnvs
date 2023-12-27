import torch.nn as nn
import torch

class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.feature_extractor = nn.Sequential(nn.Conv3d(1, 8, kernel_size=5, stride=2, padding=2),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.Flatten())
        self.net = nn.Sequential(nn.Linear(46,256),
                                 nn.ReLU(),
                                 nn.Linear(256,128),
                                 nn.ReLU(),
                                 nn.Linear(128,7),
                                 nn.Tanh())
    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        voxel = space["s3d"]
        pose = space["pose"]
        feature = self.feature_extractor(voxel)
        feature_pose = torch.cat([feature,pose],dim=1)
        return self.net(feature_pose), {}
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.feature_extractor = nn.Sequential(nn.Conv3d(1, 8, kernel_size=5, stride=2, padding=2),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.MaxPool3d(4,stride=2),
                                               nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                                               nn.ReLU(),
                                               nn.Flatten())
        self.net = nn.Sequential(nn.Linear(53,128),
                                 nn.ReLU(),
                                 nn.Linear(128,64),
                                 nn.ReLU(),
                                 nn.Linear(64,1))
    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        voxel = space["s3d"]
        pose = space["pose"]
        feature = self.feature_extractor(voxel)
        feature_pose = torch.cat([feature,pose],dim=1)
        return self.net(torch.cat([feature_pose, inputs["taken_actions"]], dim=1)), {}