# this is test file for dooropening and does not work

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask


class DoorTest(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
