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
from isaacgymenvs.tasks.utils.robot_helpers.robot_helpers.spatial import Transform, Rotation

import torch
import cv2
import time


# Quat : (x, y, z, w)

class DoorHook(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.max_episode_length = 300 # 300
        
        self.timer = None
        self.n = 0

        # door size scale for sim2real
        self.door_scale_param = 0.55
        self.door_scale_rand_param = 0.2

        # rand param for action scales
        self.action_scale_base = 0.025 # base
        self.action_scale_rand = 0.002

        # rand param for start
        self.start_pos_noise_scale = 0.5 # 0.5 
        self.start_rot_noise_scale =  0.25 # 0.25

        ############################################################
        # for test
        self.start_pos_noise_scale = 0.0
        self.start_rot_noise_scale =  0.0
        self.door_scale_rand_param = 0.0

        self.door_scale_rand_param = 0.0
        ############################################################

        # reward parameters
        self.open_reward_scale = 100.0
        self.handle_reward_scale = 75.0
        self.dist_reward_scale = 5.0
        self.o_dist_reward_scale = 1.0

        self.action_penalty_scale = 0.1 # 0.01

        self.debug_viz = False

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.aggregate_mode = 3        

        # set camera properties for realsense now : 435 [0.18, 3.0] and 405 [0.07, 0.5]
        self.camera_props = gymapi.CameraProperties()
        # after cropping
        self.img_crop_width =  64
        self.img_crop_height = 48
        # before cropping 
        self.camera_props.width = 92
        self.camera_props.height = 70
        self.depth_min = -3.0 
        self.depth_max = -0.1 # -0.07

        self.camera_props.enable_tensors = True # If False, d_img process doesnt work

        # pixel noise factor
        self.pixel_noise_factor = 0.85



        # set observation space and action space
        self.cfg["env"]["numObservations"] = 6 + self.img_crop_height*self.img_crop_width # TODO to be changed
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
        self.hand_default_dof_pose_mid = to_torch([0, 0, 1.0, 0, 0, 0], device=self.device)

        ############################################################################
        # for test
        self.hand_default_dof_pose_mid = to_torch([0, 0, 0.5, 0.7854, 0.7854, 0.7854])
        self.hand_default_dof_pose_mid = to_torch([0, 0, 0.5, 3.141592, 0, 0])
        self.hand_default_dof_pose_mid = to_torch([-0.2, -0.1, 0.5, 3.141592, 0, 0])

        ############################################################################

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # (num_envs*num_actors, 8, 2)

        self.hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_hand_dofs]  # (num_envs, 6, 2)
        # print(self.hand_dof_state.shape)
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_pos_prev = torch.zeros_like(self.hand_dof_pos, device=self.device)
        self.hand_dof_vel = self.hand_dof_state[..., 1]
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_dofs:] # (num_envs, 2, 2)
        # print(self.door_dof_state.shape)
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_pos_prev = torch.zeros_like(self.door_dof_pos, device=self.device)     

        self.hand_o_dist = None

        self.action_scale_vec = torch.zeros(self.num_envs, 1, device=self.device)

        self.door_dof_vel = self.door_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.hand_pose_world = torch.zeros(self.num_envs, 7, device=self.device)
        self.num_bodies = self.rigid_body_states.shape[1]
        # print(self.num_bodies)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    
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
        hand_asset_file = 'urdf/door_test/v2_hook_hand.urdf' # rz ry rx
        # hand_asset_file = 'urdf/door_test/hook_test.urdf'
        door_1_asset_file = 'urdf/door_test/door_1_wall.urdf'
        door_2_asset_file = 'urdf/door_test/door_2_wall.urdf'
        door_1_inv_asset_file = 'urdf/door_test/door_1_inv_wall.urdf'
        door_2_inv_asset_file = 'urdf/door_test/door_2_inv_wall.urdf'
        
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
        asset_options.collapse_fixed_joints = True
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
        self.handle_torque_tensor[:,7] = -10

        # print('----------------------------------------------- num properties ----------------------------------------')
        # print("num hand bodies: ", self.num_hand_bodies)
        # print("num hand dofs: ", self.num_hand_dofs)
        # print("num door bodies: ", self.num_door_bodies)
        # print("num door dofs: ", self.num_door_dofs)
        # print('----------------------------------------------- num properties ----------------------------------------')

        # set hand dof properties
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        for i in range(self.num_hand_dofs):
            hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # hand_dof_props['hasLimits'][i] = False
                # print(f'############### feed back ####################\n{hand_dof_props}')
            hand_dof_props['stiffness'][i] = hand_dof_stiffness[i]
            # hand_dof_props['lower'][i] = -2
            # hand_dof_props['upper'][i] = 2
            hand_dof_props['damping'][i] = hand_dof_damping[i]

            hand_dof_props['effort'][i] = 400
        print(hand_dof_props)

        # self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        # self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)
        # self.hand_dof_speed_scales = torch.ones_like(self.hand_dof_lower_limits)

        # set door dof properties
        door_dof_props = self.gym.get_asset_dof_properties(door_1_asset)
    
        # start pose
        hand_start_pose = gymapi.Transform()
        # start pose for hand
        # hand_start_pose.p = gymapi.Vec3(0.4315, -0.0213, 0.5788) # initial position of the robot # (0.4315, -0.0213, 0.5788) on UR3 in this branch
        # hand_start_pose.r = gymapi.Quat(0.0315, 0.0032, -0.9995, -0.0031)

        # origin of the urdf-robot
        hand_start_pose.p = gymapi.Vec3(0.00, 0.00, 0.0) # initial position of the robot # (0.4315, -0.0213, 0.5788) on UR3 in this branch
        hand_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)


        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(-0.75, 0.0, 0.0)
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

        # handles definition : index
        self.hook_finger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, "hook_finger")

        # rigid body index
        # self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, 'ee_rx_link')
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, 'hand_base')

        self.robot_base = self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, 'base_link')

        self.door_handle = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "door_handles")
        # print('------------self.door_handle',self.door_handle)
        self.init_data()

    def init_data(self): # NOT SURE NEED
        # hand information
        hand_pose = self.gym.get_rigid_transform(self.envs[0], self.hand_handle) # hand pose in world cordinate

        # jacobian 
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, 'hook_hand')
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor) # shape : (num_envs, 9, 6, 6)
        self.j_hand_base_link = self.jacobian[:, self.hand_handle-1, :, :]
        print(self.j_hand_base_link.shape)

        # time.sleep(10)
                
        
    def compute_reward(self, actions): #if you edit, go to jitscripts

        self.rew_buf[:], self.reset_buf[:] = compute_hand_reward(
            self.reset_buf, self.progress_buf, self.actions, self.door_dof_pos, self.door_dof_pos_prev, self.hand_dist, self.hand_o_dist,  
            self.num_envs, 
            self.open_reward_scale, self.handle_reward_scale, self.dist_reward_scale, self.o_dist_reward_scale, self.action_penalty_scale, self.max_episode_length)
        

    def debug_camera_imgs(self):
        
        # ----------------------------------------------------------------------------------------------------------------
        # import cv2
        # # combined_img = np.zeros((3, self.camera_props.width, self.camera_props.height))
        # im_list = []

        # for j in range(self.num_envs):
        #     d_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_DEPTH)
        #     np.savetxt(f"./.test_data/d_img_{j}.csv",self.pp_d_imgs[j,...].cpu().reshape(48, 64), delimiter=',')
        #     rgb_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_COLOR)
        #     reshape = rgb_img.reshape(rgb_img.shape[0],-1,4)[...,:3]
        #     im_list.append(reshape)
        #     # cv2.imshow(f'rgb{j}', rgb_img)
        # im_all = cv2.hconcat(im_list)
        # cv2.namedWindow('hand camera images',cv2.WINDOW_GUI_NORMAL)
        # cv2.imshow('hand camera images', im_all)
        # # cv2.waitKey(200)
        # cv2.waitKey(1)
        # -----------------------------------------------------------------------------------------------------------------
        # import cv2
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from io import BytesIO
        buf = BytesIO()

        cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("depth", cv2.WINDOW_NORMAL)

        for j in range(self.num_envs):
            # d_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_DEPTH)
            # np.savetxt(f"./.test_data/d_img_{j}.csv",d_img, delimiter=',')

            rgb_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_COLOR)
            rgb_img = rgb_img.reshape(rgb_img.shape[0],-1,4)[...,:3]
            cv2.imshow('rgb', rgb_img)
            # cv2.waitKey(1)

            # torch.save(self.pp_d_imgs[0, :], f'./.test_data/pp_.d_img')
            # torch.save(self.silh_d_imgs[0,:], f'./.test_data/shape_.d_img')
            # torch.save(self.th_n_d_imgs[0,:], f'./.test_data/th_n_.d_img')
            # np.savetxt(f"./.test_data/48_64_thresh_d_img.csv",self.thresh_d_imgs[j,...].cpu().view(self.camera_props.height, self.camera_props.width), delimiter=',')
            # print(self.thresh_d_imgs.shape)
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
        
        # print('cropped, stacked', self.cropped_d_imgs.shape)

    def d_img_pixel_noiser(self):

        noise_filter = self.th_n_d_imgs * torch.rand_like(self.th_n_d_imgs)
        self.th_n_d_imgs = torch.where(noise_filter > self.pixel_noise_factor, 0, self.th_n_d_imgs)


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

        self.th_n_d_imgs = torch.where(self.th_n_d_imgs > 1.0, 0, self.th_n_d_imgs)
        print('thresh_norm', 'max', torch.max(self.th_n_d_imgs), 'min', torch.min(self.th_n_d_imgs))
        # print('thresh_norm all', self.th_n_d_imgs)

        self.d_img_pixel_noiser()

        # dist_d_imgs = (self.d_imgs - self.depth_min)/(self.depth_max - self.depth_min + 1e-8)
        # dist_d_imgs[torch.where(torch.logical_or(self.d_imgs < self.depth_min, self.d_imgs > self.depth_max))] = -1.0 # replaced from -1


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
        self.debug_camera_imgs()

        #apply door handle torque_tensor as spring actuation
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.handle_torque_tensor))

        # hook finger rigid body states
        hook_pos = self.rigid_body_states[:, self.hook_finger_handle][:, 0:3] # hook finger position
        hook_rot = self.rigid_body_states[:, self.hook_finger_handle][:, 3:7] # hook finger orientation
        # hand finger rigid body states
        self.hand_pos_world = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot_world = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        self.hand_pose_world = self.rigid_body_states[:, self.hand_handle][:, 0:7]
        # hand_rot_world_euler = self.quat_to_euler(self.hand_rot_world.view(4,1))
        
        # print('hand_pose_world', self.hand_pose_world)
        # print('hand_rot_world_euler', self.quat_to_euler_tensor(self.hand_rot_world))
        # print('hand_pos_world : ',self.hand_pos_world )
        # print('hand_rot_world_euler_zyx : ', hand_rot_world_euler)

        # door handle rigid body states 
        door_handle_pos = self.rigid_body_states[:, self.door_handle][:, 0:3]
        self.hand_dist = torch.norm(door_handle_pos - hook_pos, dim = 1)
        # print(self.hand_dof_pos)
        # print(self.hand_dof_pos[:,3:])
        self.hand_o_dist = torch.norm(self.hand_dof_pos[:,3:], dim = -1)
        # print(self.hand_o_dist)
        dof_pos_dt = self.hand_dof_pos - self.hand_dof_pos_prev

        # print('dof_pos', self.hand_dof_pos)
        # print(hook_pos)
        # print(self.hand_dof_vel)
        # fake_dof_vel = dof_pos_dt/self.dt
        # print(fake_dof_vel)
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_dofs:] # (num_envs, 2, 2)
        self.door_dof_pos = self.door_dof_state[..., 0] # shape : (num_envs, 2)
        
        self.obs_buf = torch.cat((dof_pos_dt, self.pp_d_imgs), dim = -1) # TODO

        return self.obs_buf    
    
    # def quat_to_euler(self, quat): # one env
    #     quat = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
    #     euler = quat.to_euler_zyx() # can be changed xyz etc...

    #     euler_tensor = torch.tensor([euler[0], euler[1], euler[2]])

    #     euler_tensor = torch.nan_to_num(euler_tensor, nan=0)
    #     return euler_tensor
    
    def quat_to_euler_tensor(self, quat_tensor):
        euler_tensor = torch.stack([
            torch.tensor(gymapi.Quat(quat[0], quat[1], quat[2], quat[3]).to_euler_zyx()).to(self.device)
            for quat in quat_tensor.cpu().numpy()
        ])
        # print(euler_tensor)
        euler_tensor = torch.nan_to_num(euler_tensor, nan=0.0).to(quat_tensor.device)

        return euler_tensor
    
    def quat_from_euler_tensor(self, euler_tensor):
        # quat_tensor = torch.stack([
        #     torch.tensor([gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2]).x,
        #                 gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2]).y,
        #                 gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2]).z,
        #                 gymapi.Quat.from_euler_zyx(euler[0], euler[1], euler[2]).w])
        #     for euler in euler_tensor.cpu().numpy()
        # ], dim=0).to(euler_tensor.device)

        quat_tensor = torch.zeros(self.num_envs, 4, device='cuda')
        for i in range(self.num_envs):
            quat_i = gymapi.Quat.from_euler_zyx(euler_tensor[i,0], euler_tensor[i,1], euler_tensor[i,2])
            quat = torch.tensor([quat_i.x, quat_i.y, quat_i.z, quat_i.w], device='cuda')
            quat_tensor[i, :] = quat

        return quat_tensor        
    
    def reset_idx(self, env_ids):
        # print(env_ids)
        # reset hand ： tensor_clamp from torch_jit utils action dimension limitations
        # -0.25 - 0.25 noise
        # with no limit
        rand_pos = -1 * torch.rand(len(env_ids), 3, device=self.device)
        rand_rot = -1 * torch.rand(len(env_ids), 3, device=self.device)
        rand_pos += 0.5
        rand_rot += 0.5
        # rand_rot[:,1] *= 0.5 # smallen pitch rand
        rand_pos = rand_pos * self.start_pos_noise_scale
        rand_rot = rand_rot * self.start_rot_noise_scale

        pos = torch.zeros(env_ids.shape[0], 6).to(self.device)

        # # ----------------------- both side 
        # left_mask = (env_ids % 2 == 0)
        # right_mask = ~left_mask

        # left_default_pos = self.hand_default_dof_pos_left.unsqueeze(0).expand(len(env_ids), -1)
        # right_default_pos = self.hand_default_dof_pos_right.unsqueeze(0).expand(len(env_ids), -1)

        # pos[left_mask] = left_default_pos[left_mask] + torch.cat([rand_pos[left_mask], rand_rot[left_mask]], dim=-1)
        # pos[right_mask] = right_default_pos[right_mask] + torch.cat([rand_pos[right_mask], rand_rot[right_mask]], dim=-1)

        
        # print(f'Left count: {left_mask.sum().item()}, Right count: {right_mask.sum().item()}')


        # ------------------------ mid
        pos = self.hand_default_dof_pose_mid.unsqueeze(0) + torch.cat([rand_pos , rand_rot], dim=-1)
        
        # # # ------------------------ left 
        # pos = self.hand_default_dof_pos_left.unsqueeze(0) + torch.cat([rand_pos, rand_rot], dim=-1)
        # with limit
        # pos = tensor_clamp(
        #     self.hand_default_dof_pos_left.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 0.5),
        #     self.hand_dof_lower_limits, self.hand_dof_upper_limits)            

        self.hand_dof_pos[env_ids, :] = pos
        self.hand_dof_vel[env_ids, :] = torch.zeros_like(self.hand_dof_vel[env_ids])
        self.dof_targets[env_ids, :self.num_hand_dofs] = pos

        # reset door dof state
        self.door_dof_state[env_ids, :] = torch.zeros_like(self.door_dof_state[env_ids])
        self.door_dof_pos_prev[env_ids, :] = torch.zeros_like(self.door_dof_pos_prev[env_ids])       
        self.hand_dof_pos_prev[env_ids, :] = torch.zeros_like(self.hand_dof_pos_prev[env_ids])       

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

        
    def pre_physics_step(self, actions): # self.gym.set_dof_target_tensor()

        self.actions = self.action_scale_vec * actions.clone().to(self.device)
        self.actions = self.zero_actions() # action becomes [0, 0, 0, 0, 0, 0]

        # self.actions[:,1] = 0.01
        # self.actions[:,2] = 0.01
        # self.actions[:,5] = 0.01
        # print('self.actions',self.actions*self.action_scale*self.dt)

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # get informations to transform and action
        jacobian = self.j_hand_base_link # J
        # P_world_t = self.hand_pose_world # P^world_t

        p_world_t = self.hand_pose_world[:, 0:3]
        q_world_t = self.hand_pose_world[:, 3:7]
        q_world_t_euler = self.quat_to_euler_tensor(q_world_t)

        # P_hand_goal = P_world_t + A^world_t

        # # ############### test ############################### test ########################
        # # define p
        # p_world_t = torch.zeros_like(p_world_t, device=self.device)
        # p_world_t[:,...] = torch.tensor([7.4760e-08, -7.4746e-11,  5.0001e-01])

        # q_world_t = torch.zeros_like(q_world_t, device=self.device)
        # q_world_t[:,...] = torch.tensor([1.1155e-04,  9.2377e-01,  2.7060e-01,  2.7096e-01])
        # q_world_t_euler = self.quat_to_euler_tensor(q_world_t)
        # # ###################################################################

        # ##### calculate p' in world cordinate
        p_world_goal, q_world_goal_euler = transform_hand_to_world_add_action(p_world_t, q_world_t_euler, self.actions)
        q_world_goal = self.quat_from_euler_tensor(q_world_goal_euler)

        # print(self.actions)
        # q_world_goal = torch.zeros_like(q_world_goal, device=self.device) # TODO
        # -----------------------------------------------------------------
        
        # 正しいコード
        # d_dof = ik(jacobian, p_world_t, q_world_t, p_world_goal, q_world_goal, 0.05) # (self.num_envs, 6)

        d_dof_pos_test = ik(jacobian, p_world_t, q_world_t, p_world_goal, q_world_goal, 0.05)
        ###################################################################
        # # test
        # d_dof = torch.zeros_like(d_dof)
        # d_dof[:,5] = 0.1 # u
        # print('p_prime', d_dof)
        # # -----------------------------------------------------------------
        # 正しいコード -----------------------
        # targets = self.dof_targets[:, :self.num_hand_dofs] + d_dof

        targets = torch.zeros(self.num_envs, 6)
        targets[:,...] = self.dof_targets[:, :self.num_hand_dofs] + d_dof_pos_test
        # print('p_prime', targets) ############################


        ###################################################################

        # targets = 0.5*torch.ones_like(targets, device=self.device)



        # -----------------------------------------------------------------
        # ----------- without clamp limit ----------------------------------
        self.dof_targets[:, :self.num_hand_dofs] = targets 
        # ------------------------------------------------------------------
        # self.gym.set_dof_position_target_tensor(self.sim,
        #                                         gymtorch.unwrap_tensor(self.dof_targets)) # ハンドを urdf 座標系でp'動かすためのコード


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.door_dof_pos_prev = self.door_dof_pos.clone()
        
        self.hand_dof_pos_prev = self.hand_dof_pos.detach().clone()
        # print('prev_pos:',self.hand_dof_pos_prev)

    
def euler_to_rotation_matrix(euler):
    """
    Convert Euler angles to rotation matrix. The rotation order is ZYX.
    """
    cz = torch.cos(euler[:, 2])
    sz = torch.sin(euler[:, 2])
    cy = torch.cos(euler[:, 1])
    sy = torch.sin(euler[:, 1])
    cx = torch.cos(euler[:, 0])
    sx = torch.sin(euler[:, 0])

    zeros = torch.zeros_like(cz)
    ones = torch.ones_like(cz)

    Rz = torch.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=1).reshape(-1, 3, 3)
    Ry = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=1).reshape(-1, 3, 3)
    Rx = torch.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=1).reshape(-1, 3, 3)

    return Rz @ Ry @ Rx

# def quat_to_euler_batch(quat_tensor):
#     """
#     Convert a batch of quaternions to Euler angles (ZYX order).
#     """
#     euler_tensor = torch.stack([
#         torch.tensor(gymapi.Quat(quat[0], quat[1], quat[2], quat[3]).to_euler_zyx())
#         for quat in quat_tensor.cpu().numpy()
#     ])
#     euler_tensor = torch.nan_to_num(euler_tensor, nan=0.0).to(quat_tensor.device)

#     return euler_tensor


# def transform_hand_to_world_add_action(p_world_t, q_world_t_euler, actions):
#     """
#     Transform hand coordinates to world coordinates after applying the action.

#     Parameters:
#     p_world_t (torch.Tensor): Hand positions in world coordinates at time t (shape: (num_envs, 3)).
#     q_world_t_euler (torch.Tensor): Hand orientations in world coordinates at time t (shape: (num_envs, 3)).
#     action (torch.Tensor): Actions in hand coordinates (shape: (num_envs, 6)).

#     Returns:
#     None: Updates self.p_world_goal and self.q_world_goal.
#     """
#     # Compute rotation matrix from Euler angles
#     R_world_t = euler_to_rotation_matrix(q_world_t_euler)

#     # Extract position and orientation change from action
#     delta_pos_hand = actions[:, :3]
#     delta_euler_hand = actions[:, 3:]

#     # Compute new position in world coordinates
#     p_world_goal = p_world_t + torch.bmm(R_world_t, delta_pos_hand.unsqueeze(-1)).squeeze(-1)

#     # Compute new orientation in world coordinates
#     # q_world_goal_euler = q_world_t_euler + delta_euler_hand # TODO not proper
#     q_world_goal_euler = q_world_t_euler + torch.bmm(R_world_t, delta_euler_hand.unsqueeze(-1)).squeeze(0-1)

#     return p_world_goal, q_world_goal_euler


def transform_hand_to_world_add_action(p_world_t, q_world_t_euler, actions):
    """

    for test, so after finish, delete this.
    """
    # Compute rotation matrix from Euler angles
    R_world_t = euler_to_rotation_matrix(q_world_t_euler)

    # Extract position and orientation change from action
    delta_pos_hand = actions[:, :3]
    delta_euler_hand = actions[:, 3:]

    # Compute new position in world coordinates
    p_world_goal = p_world_t + torch.bmm(R_world_t, delta_pos_hand.unsqueeze(-1)).squeeze(-1)
    print('in the function : p', p_world_t, p_world_goal)

    # Compute new orientation in world coordinates
    # q_world_goal_euler = q_world_t_euler + delta_euler_hand # TODO not proper
    q_world_goal_euler = q_world_t_euler + torch.bmm(R_world_t, delta_euler_hand.unsqueeze(-1)).squeeze(0-1)
    print('in the function : q', q_world_t_euler, q_world_goal_euler)

    return p_world_goal, q_world_goal_euler





#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    reset_buf, progress_buf, actions, door_dof_pos, door_dof_pos_prev, hand_dist, hand_o_dist, num_envs, open_reward_scale, handle_reward_scale, dist_reward_scale, o_dist_reward_scale,
    action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    # print(open_reward_scale, handle_reward_scale, dist_reward_scale, action_penalty_scale)
    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(-1*actions ** 2, dim=-1) * action_penalty_scale
    # handle_reward=torch.zeros(1,num_envs)
    # open_reward = door_dof_pos[:,0] * door_dof_pos[:,0] * open_reward_scale
    # open_reward = (door_dof_pos[:,0] - door_dof_pos_prev[:,0]) * open_reward_scale
    open_reward = door_dof_pos[:,0] * open_reward_scale    # additional reward to open
    handle_reward = door_dof_pos[:,1] * handle_reward_scale

    hand_dist_thresh = torch.where(hand_dist < 0.15, torch.zeros_like(hand_dist), hand_dist)

    # dist_reward = -1 * hand_dist * dist_reward_scale # no thlesh
    dist_reward = -1 * hand_dist_thresh * dist_reward_scale

    o_dist_reward = -1 * hand_o_dist * o_dist_reward_scale
    # print(o_dist_reward)

    # dist_reward_no_thresh = -1 * (hand_dist + torch.log(hand_dist + 0.005)) * dist_reward_scale
    # dist_reward_no_thresh = -1 * hand_dist * dist_reward_scale
    # print(hand_dist.shape)
    # print('hand_o_dist.shape', hand_o_dist.shape)
    # print('----------------open_reward max:',torch.max(open_reward))
    # print('--------------handle_reward max:', torch.max(handle_reward))
    # print('----------------dist_min:', torch.min(hand_dist))
    # print('-------------action_penalty max:', torch.min(action_penalty))

    # rewards = open_reward + dist_reward_no_thresh + handle_reward + action_penalty
    rewards = open_reward + dist_reward + o_dist_reward + handle_reward + action_penalty
    # rewards = open_reward + dist_reward + handle_reward + action_penalty
    # success reward
    # rewards = torch.where(door_dof_pos[:,0] > 1.55, rewards + 1000, rewards)

    # print('-------------------door_hinge_max :', torch.max(door_dof_pos[:,0]), 'door_hinge_min :', torch.min(door_dof_pos[:,0]))
    # print('-------------------door_handle_max :', torch.max(door_dof_pos[:,1]), 'door_handle_min :', torch.min(door_dof_pos[:,1]))
    # print('----------------------rewards_max :', torch.max(rewards), 'rewards_min :',torch.min(rewards))

    reset_buf = torch.where(door_dof_pos[:, 0] >= 1.56, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

@torch.jit.script
def compute_d_imgs(d_imgs, depth_min, depth_max, replace_val): # wasnt better than normal function

    # type: (Tensor, float, float, float) -> Tensor 
    condition = torch.logical_or(d_imgs < depth_min, d_imgs > depth_max)
    dist_d_imgs = (d_imgs - depth_min)/(depth_max - depth_min)
    dist_d_imgs = torch.where(condition, replace_val, dist_d_imgs)
    return dist_d_imgs 












        


