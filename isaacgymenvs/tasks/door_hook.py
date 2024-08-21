# this is test file for dooropening and works but uncompleted


import numpy as np
import os
# import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from skrl.utils.isaacgym_utils import ik

import torch
import cv2
import time
import datetime
import csv


# Quat : (x, y, z, w)

class DoorHook(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.max_episode_length = 300 # 300
        
        self.timer = None
        self.env_episodes = None
        self.begin_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.n = 0

        # door size scale for sim2real
        self.door_scale_param = 0.8
        self.door_scale_rand_param = 0.1

        # rand param for action scales
        self.action_scale_base = 0.045 # base # 0.025?
        self.action_scale_rot_ratio = 1.0
        self.action_scale_rand = 0.0005 # noise

        # rand param for start
        self.start_pos_noise_scale = 0.25 # 0.5 
        self.start_rot_noise_scale =  0.2 # 0.25

        # reward parameters
        self.open_reward_scale = 100.0
        self.handle_reward_scale = 50.0
        self.dist_reward_scale = 5.0
        self.o_dist_reward_scale = 1.0 # TODO

        self.action_penalty_scale = 0.05 # 0.01 # TODO

        self.distance_thresh = 0.04
        self.hook_handle_reset_dist = 2.0

        # door handle torque
        self.handle_torque = 12.5

        self.debug_viz = False

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.aggregate_mode = 3

        # set camera properties for realsense now : 435 [0.18, 3.0] and 405 [0.07, 0.5]
        self.camera_props = gymapi.CameraProperties() # TODO camera parameters shoud be changed
        # after cropping
        self.img_crop_width =  64
        self.img_crop_height = 48
        # before cropping 
        self.camera_props.width = 92
        self.camera_props.height = 70
        self.depth_min = -3.0 
        self.depth_max = -0.07 # -0.07

        self.camera_props.enable_tensors = True # If False, d_img process doesnt work


        # set observation space and action space
        self.cfg["env"]["numObservations"] = 9 + 9 + 9 + 3 + self.img_crop_height*self.img_crop_width
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # initialize tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.hand_default_dof_pose_mid = to_torch([0, 0, 0.8, 3.141592653, 0, 0], device=self.device)

        ############################################################################
        # for test
        # self.hand_default_dof_pose_mid = to_torch([0, 0, 0.5, 0.7854, 0.7854, 0.7854])
        # self.hand_default_dof_pose_mid = to_torch([0, 0, 0.5, 4.6, 0.5, -1.2])
        # self.hand_default_dof_pose_mid = to_torch([-0.2, -0.1, 0.5, 3.141592, 0, 0]) # to door
        # self.hand_default_dof_pose_mid = to_torch([0.2, 0.3, 1.0, 0.7, 0.7, 0.7])

        ############################################################################

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # (num_envs*num_actors, 8, 2)

        self.hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_hand_dofs]  # (num_envs, 6, 2)
        # print(self.hand_dof_state.shape)
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_pose_world_prev = torch.zeros(self.num_envs, 7, device=self.device)
        self.hand_dof_vel = self.hand_dof_state[..., 1]
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_dofs:] # (num_envs, 2, 2)
        # print(self.door_dof_state.shape)
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_pos_prev = torch.zeros_like(self.door_dof_pos, device=self.device)

        self.hook_handle_dist = torch.zeros(self.num_envs, 1, device=self.device)
        self.R_diff_norm = torch.zeros(self.num_envs, 1, device=self.device)
        self.hook_dsr_init_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.hook_dsr_init_rmat = torch.zeros(self.num_envs, 3, 3, device=self.device)

        self.action_scale_vec = torch.zeros(self.num_envs, 1, device=self.device)

        self.door_dof_vel = self.door_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.hand_pose_world = torch.zeros(self.num_envs, 7, device=self.device)
        self.door_fake_link_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.num_bodies = self.rigid_body_states.shape[1]
        # print(self.num_bodies)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        self.hook_dsr_init_rmat = quaternion_to_rotation_matrix(self.rigid_body_states.clone().detach()[:, self.hook_finger_dsr_pose][:, 3:7])

        # for statics
        self.statistics = []
        self.knob_range = 0.01
        self.hinge_range = 0.01
        self.success_range = 1.0472 # opening 60 deg
        self.reaching_swich = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.opening_swich = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_swich = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
    
        self._create_ground_plane()
        # print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        hand_asset_file = 'urdf/door_test/v4_hook_hand.urdf' 
        door_1_asset_file = 'urdf/door_test/door_1_wall.urdf'
        door_2_asset_file = 'urdf/door_test/door_2_wall.urdf'
        door_1_inv_asset_file = 'urdf/door_test/door_1_wall.urdf'
        door_2_inv_asset_file = 'urdf/door_test/door_2_wall.urdf'
        
        # load hand asset
        asset_options = gymapi.AssetOptions()
        vh_options = gymapi.VhacdParams()

        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.vhacd_enabled = True # if True, accurate collision enabled
        vh_options.max_convex_hulls = 10000
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.1
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)

        # load door asset
        asset_options.flip_visual_attachments = False
        # asset_options.collapse_fixed_joints = True
        asset_options.collapse_fixed_joints = False

        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        door_1_asset = self.gym.load_asset(self.sim, asset_root, door_1_asset_file, asset_options)
        door_2_asset = self.gym.load_asset(self.sim, asset_root, door_2_asset_file, asset_options)
        door_1_inv_asset = self.gym.load_asset(self.sim, asset_root, door_1_inv_asset_file, asset_options)
        door_2_inv_asset = self.gym.load_asset(self.sim, asset_root, door_2_inv_asset_file, asset_options)
        door_assets = [door_1_asset, door_2_asset, door_1_inv_asset, door_2_inv_asset]

        hand_dof_stiffness = to_torch([500, 500, 500, 500, 500, 500], dtype=torch.float, device=self.device)
        hand_dof_damping = to_torch([10, 10, 10, 10, 10, 10], dtype=torch.float, device=self.device)

        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_1_asset)
        self.num_door_dofs = self.gym.get_asset_dof_count(door_1_asset)

        # torque tensor for door handle
        self.handle_torque_tensor = torch.zeros([self.num_envs, self.num_hand_dofs+self.num_door_dofs], dtype=torch.float, device=self.device)
        self.handle_torque_tensor[:,7] = -1 * self.handle_torque

        # set hand dof properties
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        for i in range(self.num_hand_dofs):
            hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            hand_dof_props['stiffness'][i] = hand_dof_stiffness[i]
            hand_dof_props['lower'][i] *= 2.0
            hand_dof_props['upper'][i] *= 2.0
            hand_dof_props['damping'][i] = hand_dof_damping[i]

            hand_dof_props['effort'][i] = 500
            
        print(hand_dof_props)

        # ---------------- set dof limits from urdf -----------------------------------------------
        # self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        # self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)
        # self.hand_dof_speed_scales = torch.ones_like(self.hand_dof_lower_limits)

        # set door dof properties
        door_dof_props = self.gym.get_asset_dof_properties(door_1_asset)
    
        # start posese
        hand_start_pose = gymapi.Transform()

        # origin of the urdf-robot
        hand_start_pose.p = gymapi.Vec3(1.0, 0.00, 0.1) # robot base position  # (0.4315, -0.0213, 0.5788) on UR3 in this branch
        hand_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0) # robot base rotation


        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(-0.1, 0.0, 0.0)
        door_start_pose.r = gymapi.Quat(0, 0, 0, 1)

        # compute aggregate size
        num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        num_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        num_door_bodies = self.gym.get_asset_rigid_body_count(door_1_asset)
        num_door_shapes = self.gym.get_asset_rigid_shape_count(door_1_asset)

        max_agg_bodies = num_hand_bodies + num_door_bodies
        max_agg_shapes = num_hand_shapes + num_door_shapes

        # camera pose setting
        camera_tf = gymapi.Transform()
        camera_tf.p = gymapi.Vec3(0.1, 0, 0.05)
        camera_tf.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.camera_props.enable_tensors = True # when Vram larger

        print('#############################################################################################################')
        print(f'num_hand_bodies : {num_hand_bodies}, num_hand_shapes : {num_hand_shapes}, \nnum_door_bodies : {num_door_bodies}, num_door_shapes : {num_door_shapes}')
        print('#############################################################################################################')

        self.hands = []
        self.doors = []
        self.envs = []
        self.camera_handles = []
        
        door_asset_count = 0
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot hand actor name as "robot_hand"
            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hook_hand", i, 0, 0)
                                                                                                # ↑self collision ON
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)

            # create door actors # all doors ------------------------------------------
            if door_asset_count == 3:
                door_actor = self.gym.create_actor(env_ptr, door_assets[door_asset_count], door_start_pose, "door", i, 0, 0)
                door_asset_count = 0
            else:
                door_actor = self.gym.create_actor(env_ptr, door_assets[door_asset_count], door_start_pose, "door", i, 0, 0)
                door_asset_count += 1
            # -------------------------------------------------------------------------
                
            # # only pull door ---------------------------------------------------------
            # if i % 2 == 0:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[0], door_start_pose, "door", i, 0, 0)
            # else:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[1], door_start_pose, "door", i, 0, 0)
            # # -------------------------------------------------------------------------
                
            # # only rihgt hinge ---------------------------------------------------------
            # if i % 2 == 0:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[0], door_start_pose, "door", i, 0, 0)
            # else:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[2], door_start_pose, "door", i, 0, 0)
            # # -------------------------------------------------------------------------
                
            self.gym.set_actor_dof_properties(env_ptr, door_actor, door_dof_props)

            # setting for friction
            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
            door_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, door_actor)
            hand_shape_props[3].friction = 0.1 # index of hook body
            hand_shape_props[3].torsion_friction = 0.1
            door_shape_props[13].friction = 0.1 # index of door handle
            door_shape_props[13].torsion_friction = 0.1

            self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, door_actor, door_shape_props)

            #door size randomization
            self.gym.set_actor_scale(env_ptr, door_actor, self.door_scale_param + (torch.rand(1) - 0.5) * self.door_scale_rand_param)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.hands.append(hand_actor)
            self.doors.append(door_actor)

            camera_handle = self.gym.create_camera_sensor(self.envs[i], self.camera_props)
            self.camera_handles.append(camera_handle)
            # camera_mnt = self.gym.find_actor_rigid_body_handle(self.envs[i], hand_actor, "ee_rz_link")
            camera_mnt = self.gym.find_actor_rigid_body_handle(self.envs[i], hand_actor, "ee_rx_link")

            self.gym.attach_camera_to_body(camera_handle, self.envs[i], camera_mnt, camera_tf, gymapi.FOLLOW_TRANSFORM)

        # hook_finger handles definition
        self.hook_finger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, "hook_finger")

        # rigid body index to get tf
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, 'hand_base')
        self.robot_base = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, 'base_link')
        self.door_handle = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "door_handles")
        self.door_fake_link = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "door_fake_link")
        self.hook_finger_dsr_pose = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "dsr_pose")


        self.init_data()

    def init_data(self): 
        # jacobian 
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, 'hook_hand')
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor) # shape : (num_envs, 9, 6, 6)
        self.j_hand_world = self.jacobian[:, self.hand_handle-1, :, :]

        # used for rotation matrix
        self.batch_eye = torch.stack([torch.eye(3,3) for i in range(self.num_envs)]).to(self.device)

        self.env_episodes = torch.zeros(self.num_envs, device=self.device, dtype=torch.int16)

                
        
    def compute_reward(self, actions): #if you edit, go to jitscripts

        self.rew_buf[:], self.reset_buf[:] = compute_hand_reward(
            self.reset_buf, self.progress_buf, self.actions, self.door_dof_pos, self.hook_handle_reset_dist, self.hook_handle_dist, self.distance_thresh, self.R_diff_norm,  
            self.num_envs, 
            self.open_reward_scale, self.handle_reward_scale, self.dist_reward_scale, self.o_dist_reward_scale, self.action_penalty_scale, self.max_episode_length)
        

    def debug_camera_imgs(self): # ------- works with num_envs = 1
        
        # --------------------------- RGB in one env ----------------------------------------#
        cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)

        for j in range(self.num_envs):
            # d_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_DEPTH)
            # np.savetxt(f"./.test_data/d_img_{j}.csv",d_img, delimiter=',')

            rgb_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_COLOR)
            rgb_img = rgb_img.reshape(rgb_img.shape[0],-1,4)[...,:3]
            cv2.imshow('rgb', rgb_img)
            cv2.waitKey(1)
        # -----------------------------------------------------------------------------------#

        # --------------------------- DEPTH in one env ----------------------------------------#
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from io import BytesIO
        buf = BytesIO()

        plt.axis('off')
        plt.imshow(self.pp_d_imgs[0].view(self.img_crop_height, self.img_crop_width).to('cpu').detach().numpy().copy(), cmap='coolwarm_r', norm=Normalize(vmin=0, vmax=1))
        # plt.colorbar()
        plt.savefig(buf, format = 'png')
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
        buf.close()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.namedWindow("ESDEpth", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('ESDEpth', img)
        cv2.waitKey(1)

        # plt.colorbar()
        plt.close()
        # -------------------------------------------------------------------------------------#
    def d_img_cropper(self):

        start_y = (self.camera_props.height - self.img_crop_height) // 2
        start_x = (self.camera_props.width - self.img_crop_width) // 2
        # print(start_y, start_x)
        # print('before cropping', self.d_imgs.shape)
        self.cropped_d_imgs = torch.stack([
                self.d_imgs[i, ...]
                    .view(self.camera_props.height, self.camera_props.width)[start_y:start_y + self.img_crop_height, start_x:start_x + self.img_crop_width]
                    .reshape(self.img_crop_height * self.img_crop_width)
                for i in range(self.num_envs)
            ])            
        

    def d_img_pixel_noiser(self): # TODO


        rand_tensor = torch.randn_like(self.th_n_d_imgs) + 4.75
        # rand_tensor = torch.rand_like(self.th_n_d_imgs)
        self.th_n_d_imgs = torch.where(self.th_n_d_imgs > rand_tensor, 0, self.th_n_d_imgs)


    def d_img_process(self):

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.d_imgs = torch.stack([
            gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)).view(self.camera_props.height * self.camera_props.width)
            for env, camera_handle in zip(self.envs, self.camera_handles)]).to(self.device)
        # print(torch.max(d_imgs), torch.min(d_imgs))

        self.d_img_cropper()

        self.thresh_d_imgs = torch.where(torch.logical_or(self.cropped_d_imgs <= self.depth_min, self.cropped_d_imgs >= self.depth_max), 0, self.cropped_d_imgs)
        # print('thresh_raw', torch.max(thresh_d_imgs), torch.min(thresh_d_imgs))
        # print('thresh_d_imgs shape', self.thresh_d_imgs.shape)

        self.th_n_d_imgs = (self.thresh_d_imgs - self.depth_min)/(self.depth_max - self.depth_min)

        self.th_n_d_imgs = torch.where(self.th_n_d_imgs > 1.0, 0, self.th_n_d_imgs) # change replace value after normalization
        # print('thresh_norm', 'max', torch.max(self.th_n_d_imgs), 'min', torch.min(self.th_n_d_imgs))
        # print('thresh_norm all', self.th_n_d_imgs)

        # self.d_img_pixel_noiser()

        self.silh_d_imgs = torch.stack([(self.th_n_d_imgs[i,...]  - torch.min(self.th_n_d_imgs[i,...]))/
                                        (torch.max(self.th_n_d_imgs[i,...])-torch.min(self.th_n_d_imgs[i,...]) + 1e-12) 
                                        for i in range(self.num_envs)])
        # print('silh',torch.max(self.silh_d_imgs), torch.min(self.silh_d_imgs))
        # print('silh all', self.silh_d_imgs)

        self.pp_d_imgs = 0.5*(self.th_n_d_imgs + self.silh_d_imgs)
        # print('pp',torch.max(self.pp_d_imgs), torch.min(self.pp_d_imgs))

        # self.get_d_img_dataset()

        self.gym.end_access_image_tensors(self.sim)
        # print('pp_d_img', self.pp_d_imgs)
        # print(self.dist_d_imgs[0]) # debug

    def get_d_img_dataset(self):

        for z in range(self.num_envs):
            torch.save(self.pp_d_imgs[z, :], f'../../depthnet/depth_dataset_v4/trash_{self.n}_{z}.d_img')
        self.n = self.n + 1

    def compute_observations(self): 
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.d_img_process()
        # self.debug_camera_imgs()

        #apply door handle torque_tensor as spring actuation
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.handle_torque_tensor))

        # hook finger and hook finger dsr_pose rigid body states
        hook_pose = self.rigid_body_states[:, self.hook_finger_handle][:, 0:7] # hook finger pose
        hook_dsr_pose = self.rigid_body_states[:, self.hook_finger_dsr_pose][:, 0:7] # hook finger dsr pose
        
        # hand rigid body states
        self.hand_pose_world = self.rigid_body_states[:, self.hand_handle][:, 0:7]

        # get door_fake_link_quat for state
        self.door_fake_link_quat = self.rigid_body_states[:,self.door_fake_link, 3:7]

        # hand poses
        # at t
        q_door_hand_t_quat = self.hand_pose_world[:,3:7]
        q_door_hand_t_rmat = quaternion_to_rotation_matrix(q_door_hand_t_quat)# rot_mat STATE_1 3 3*3
        # at t-1
        q_door_hand_prev_quat = self.hand_pose_world_prev[:,3:7]
        q_door_hand_prev_rmat = quaternion_to_rotation_matrix(q_door_hand_prev_quat)# rot_mat STATE_2 3*3
        d_q_door_hand_mat = torch.bmm(q_door_hand_t_rmat.clone(), q_door_hand_prev_rmat.clone().transpose(1,2)) - self.batch_eye # d_rot STATE_3_2 3*3
        # position
        p_door_hand_t = self.hand_pose_world[:,:3]
        p_door_hand_prev = self.hand_pose_world_prev[:,:3]
        d_p_door_hand = p_door_hand_t - p_door_hand_prev # pos diff STATE_3_1 3
        # norm_d_p_door_hand = d_p_door_hand / torch.sqrt(3 * self.action_scale_vec**2)

        # # normalized rot state vectors
        # norm_q_door_hand_t = q_door_hand_t / torch.sqrt(torch.tensor(3))
        # norm_q_door_hand_prev = q_door_hand_prev / torch.sqrt(torch.tensor(3))
        hand_rot_state_vector = torch.cat((q_door_hand_t_rmat.view(-1, 9), q_door_hand_prev_rmat.view(-1, 9), d_q_door_hand_mat.view(-1, 9)), dim=-1)

        # values for reward : door_dof states, hook pose diffs
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_dofs:] # (num_envs, 2, 2)
        self.door_dof_pos = self.door_dof_state[..., 0] # shape : (num_envs, 2)
        self.hook_handle_dist = torch.norm(hook_dsr_pose[:, 0:3] - hook_pose[:, 0:3], dim = 1) # hook handle distance
        R_hook_t = quaternion_to_rotation_matrix(hook_pose[:,3:7])
        R_dsr = self.hook_dsr_init_rmat
        R_diff = torch.bmm(R_hook_t.clone().transpose(1,2), R_dsr) - self.batch_eye
        self.R_diff_norm = torch.linalg.matrix_norm(R_diff)
        
        # compute state vector
        self.obs_buf = torch.cat((hand_rot_state_vector, d_p_door_hand,  self.pp_d_imgs), dim = -1)

        return self.obs_buf
    
        
    def reset_idx(self, env_ids):

        # without limit
        rand_pos = -1 * torch.rand(len(env_ids), 3, device=self.device) + 0.5
        rand_rot = -1 * torch.rand(len(env_ids), 3, device=self.device) + 0.5
        rand_pos = rand_pos * self.start_pos_noise_scale
        rand_rot = rand_rot * self.start_rot_noise_scale

        pos = torch.zeros(env_ids.shape[0], 6).to(self.device)

        # ------------------------ mid
        pos = self.hand_default_dof_pose_mid.unsqueeze(0) + torch.cat([rand_pos , rand_rot], dim=-1)
        
        self.hand_dof_pos[env_ids, :] = pos
        self.hand_dof_vel[env_ids, :] = torch.zeros_like(self.hand_dof_vel[env_ids])
        self.dof_targets[env_ids, :self.num_hand_dofs] = pos

        # reset door dof state
        self.door_dof_state[env_ids, :] = torch.zeros_like(self.door_dof_state[env_ids])
        self.door_dof_pos_prev[env_ids, :] = torch.zeros_like(self.door_dof_pos_prev[env_ids])       

        # reset hand_pose_world_prev
        self.hand_pose_world_prev[env_ids, :] = torch.zeros_like(self.hand_pose_world_prev[env_ids])       

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.action_scale_vec[env_ids] = self.action_scale_rand * (torch.rand(len(env_ids), 1, device=self.device) - 0.5) + self.action_scale_base
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.env_episodes[env_ids] += 1

        
    def pre_physics_step(self, actions): # apply action here by self.gym.set_dof_target_tensor()

        # actions = self.uni_actions() # action becomes [1, 1, 1, 1, 1, 1]
        # actions = self.zero_actions() # action [0, 0, 0, 0, 0, 0]
        # actions[:,5] = 1.0
        # actions[:,4] = 1.0
        
        self.actions = self.action_scale_vec * actions.clone().to(self.device)
        # scaling rot action
        self.actions[:,3:] = self.actions[:,3:] * self.action_scale_rot_ratio

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # get informations to transform and action
        jacobian = self.j_hand_world # J

        p_world_t = self.hand_pose_world[:, 0:3]
        q_world_t = self.hand_pose_world[:, 3:7]
        # q_world_t_euler = self.quat_to_euler_tensor(q_world_t)
        
        p_world_goal = p_world_t + self.actions[:,0:3]
        q_world_goal = quaternion_multiply(euler_to_quaternion_tensor(self.actions[:,3:]), q_world_t) # dq @ qt
        #+ quat_to_euler_tensor(self.actions[:,3:]) # TODO

        # p_world_goal, q_world_goal = transform_hand_to_world_add_action(p_world_t, q_world_t, self.actions) 
            
        d_dof = ik(jacobian, p_world_t, q_world_t, p_world_goal, q_world_goal)

        hand_dof_targets = self.dof_targets[:, :self.num_hand_dofs] + d_dof

        # ----------- without clamp limit ----------------------------------
        self.dof_targets[:, :self.num_hand_dofs] = hand_dof_targets 

        # set robot dof pose target
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_targets))


    def statistical_data_acquisition(self,env_ids):
        reaching = self.reaching_swich[env_ids].cpu().tolist()
        opening = self.opening_swich[env_ids].cpu().tolist()
        success = self.success_swich[env_ids].cpu().tolist()

        index_list = env_ids.cpu().tolist()
        statis = [[index_list[i], self.env_episodes[i].item(), reaching[i], opening[i], success[i]] for i in range(len(index_list))]
        self.statistics.append(statis)
        # Create the directory if it doesn't exist
        dir_path = 'statistic_data'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Path to the CSV file
        file_path = os.path.join(dir_path, f'RL_static_{self.begin_datetime}.csv')
        
        # Check if the file exists
        file_exists = os.path.isfile(file_path)
        
        # Open the file in append mode
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write the header if the file is being created
            if not file_exists:
                csvwriter.writerow(['env_ids', 'episode', 'reaching', 'opening', 'success'])
            
            # Write the data
            for row in statis:
                csvwriter.writerow(row)        

        self.reaching_swich[env_ids] = False
        self.opening_swich[env_ids] = False
        self.success_swich[env_ids] = False

    
    def data_acquisition(self):
        knob_range_tensor = torch.full_like(self.door_dof_pos[:, 1], self.knob_range)
        hinge_range_tensor = torch.full_like(self.door_dof_pos[:, 0], self.hinge_range)
        success_tenros = torch.full_like(self.door_dof_pos[:, 0], self.success_range)

        abs_knob_tensor = self.door_dof_pos[:, 1] - knob_range_tensor
        abs_hinge_tensor = self.door_dof_pos[:, 0] - hinge_range_tensor
        abs_success_tensor = self.door_dof_pos[:, 0] - success_tenros
                
        reaching_indices = (abs_knob_tensor >= 0).nonzero(as_tuple=True)[0]
        self.reaching_swich[reaching_indices] = True

        opening_indices = (abs_hinge_tensor >= 0).nonzero(as_tuple=True)[0]
        self.opening_swich[opening_indices] = True

        success_indices = (abs_success_tensor >= 0).nonzero(as_tuple=True)[0]
        self.success_swich[success_indices] = True


    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.data_acquisition()
        if len(env_ids) > 0:
            self.statistical_data_acquisition(env_ids)
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        # self.door_dof_pos_prev = self.door_dof_pos.clone()
        
        self.hand_pose_world_prev = self.hand_pose_world.detach().clone()
        
        # print('prev_pos:',self.hand_pose_world_prev)
        
    
def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternions to rotation matrix.
    """
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(-1, 3, 3)
    return matrix

def quat_conj(quat_tensor):
    """
    Compute the conjugate of a batch of quaternions.
    Args:
    - q: tensor of shape (num_envs, 4)
    
    Returns:
    - q_conj: tensor of shape (num_envs, 4)
    """
    q_conj = quat_tensor.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]
    return q_conj

def quat_to_euler_tensor(quat_tensor):

    # Extract individual components of the quaternions
    qx, qy, qz, qw = quat_tensor[:, 0], quat_tensor[:, 1], quat_tensor[:, 2], quat_tensor[:, 3]

    # Compute the Euler angles in z, y, x order
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_z = torch.atan2(siny_cosp, cosy_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch_y = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # roll (x-axis rotation)
    sinx_cosp = 2.0 * (qw * qx + qy * qz)
    cosx_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll_x = torch.atan2(sinx_cosp, cosx_cosp)

    euler_tensor = torch.stack((roll_x, pitch_y, yaw_z), dim=-1)

    # Handle NaNs if necessary
    euler_tensor = torch.nan_to_num(euler_tensor, nan=0.0).to(quat_tensor.device)

    return euler_tensor

def euler_to_quaternion_tensor(euler):
    """
    Convert Euler angles to quaternions.
    """
    cy = torch.cos(euler[:, 2] * 0.5)
    sy = torch.sin(euler[:, 2] * 0.5)
    cp = torch.cos(euler[:, 1] * 0.5)
    sp = torch.sin(euler[:, 1] * 0.5)
    cr = torch.cos(euler[:, 0] * 0.5)
    sr = torch.sin(euler[:, 0] * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack((qx, qy, qz, qw), dim=-1)

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    q1, q2: (num_envs, 4)
    """
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return torch.stack((x, y, z, w), dim=-1)

def transform_hand_to_world_add_action(p_world_t, q_world_t, actions):
    """
    Transform hand coordinates to world coordinates after applying the action.

    Parameters:
    p_world_t (torch.Tensor): Hand positions in world coordinates at time t (shape: (num_envs, 3)).
    q_world_t (torch.Tensor): Hand orientations in world coordinates as quaternions at time t (shape: (num_envs, 4)).
    actions (torch.Tensor): Actions in hand coordinates (shape: (num_envs, 6)).

    Returns:
    p_world_goal (torch.Tensor): Goal positions in world coordinates (shape: (num_envs, 3)).
    q_world_goal (torch.Tensor): Goal orientations in world coordinates as quaternions (shape: (num_envs, 4)).
    """

    # Convert quaternions to rotation matrices
    R_world_t = quaternion_to_rotation_matrix(q_world_t)

    # Extract position and orientation change from action
    delta_pos_hand = actions[:, :3]
    delta_euler_hand = actions[:, 3:]
    delta_quat_hand = euler_to_quaternion_tensor(delta_euler_hand)
    
    # Compute new position in world coordinates
    p_world_goal = p_world_t + torch.bmm(R_world_t, delta_pos_hand.unsqueeze(-1)).squeeze(-1)
    
    # Compute new orientation in world coordinates
    q_world_goal = quaternion_multiply(q_world_t, delta_quat_hand)
    
    return p_world_goal, q_world_goal


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward(
    reset_buf, progress_buf, actions, door_dof_pos, hook_handle_reset_dist, hook_handle_dist, distance_thresh, R_diff_norm, num_envs, open_reward_scale, handle_reward_scale, dist_reward_scale, o_dist_reward_scale,
    action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, int, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(-1*actions ** 2, dim=-1) * action_penalty_scale
    # handle_reward=torch.zeros(1,num_envs)
    # open_reward = door_dof_pos[:,0] * door_dof_pos[:,0] * open_reward_scale
    # open_reward = (door_dof_pos[:,0] - door_dof_pos_prev[:,0]) * open_reward_scale
    open_reward = door_dof_pos[:,0] * open_reward_scale    # reward to open
    # open_reward = torch.where(door_dof_pos[:,0] < 0.01745, 0, open_reward) # 1 deg thresh
    
    handle_reward = door_dof_pos[:,1] * handle_reward_scale

    # hook_handle_dist_thresh = torch.where(hook_handle_dist < distance_thresh, torch.zeros_like(hook_handle_dist), hook_handle_dist)

    # dist_reward = -1 * hook_handle_dist * dist_reward_scale # no thresh
    # dist_reward = -1 * hook_handle_dist_thresh * dist_reward_scale
    # dist_reward = (torch.exp(-20*hook_handle_dist) -1 - hook_handle_dist) * dist_reward_scale # add exp
    dist_reward = (1/((1 + hook_handle_dist**2)**2) * (1+torch.where(hook_handle_dist <= distance_thresh, 1, 0))) * dist_reward_scale # RLAfford

    o_dist_reward = -1 * R_diff_norm * o_dist_reward_scale 

    # dist_reward_no_thresh = -1 * (hook_handle_dist + torch.log(hook_handle_dist + 0.005)) * dist_reward_scale
    # dist_reward_no_thresh = -1 * hook_handle_dist * dist_reward_scale
    # print(hook_handle_dist.shape)
    # print('R_diff_norm.shape', R_diff_norm.shape)
    # print('----------------open_reward max:',torch.max(open_reward))
    # print('--------------handle_reward max:', torch.max(handle_reward))
    # print('----------------dist_min:', torch.min(hook_handle_dist))
    # print('-------------action_penalty max:', torch.min(action_penalty))

    # rewards = open_reward + dist_reward_no_thresh + handle_reward + action_penalty
    rewards = open_reward + dist_reward + o_dist_reward + handle_reward + action_penalty
    # rewards = open_reward + dist_reward + handle_reward + action_penalty # without rotation reward
    # success reward
    # rewards = torch.where(door_dof_pos[:,0] > 1.55, rewards + 1000, rewards)


    print('-------------------door_hinge_max :', torch.max(door_dof_pos[:,0]), 'door_hinge_min :', torch.min(door_dof_pos[:,0]))
    print('-------------------door_handle_max :', torch.max(door_dof_pos[:,1]), 'door_handle_min :', torch.min(door_dof_pos[:,1]))
    # print('----------------------rewards_max :', torch.max(rewards), 'rewards_min :',torch.min(rewards))

    reset_buf = torch.where(door_dof_pos[:, 0] >= 1.56, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(hook_handle_dist > hook_handle_reset_dist, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf