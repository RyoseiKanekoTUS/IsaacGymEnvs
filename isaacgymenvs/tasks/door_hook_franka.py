
import numpy as np
import os
import random
import math

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
# from base.vec_task import VecTask

import torch

from skrl.utils.isaacgym_utils import ik


class Franka_DoorHook(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # for stochastic data

        self.data_count = 0


        self.cfg = cfg
        self.n = 0
        self.max_episode_length = 300 # 300

        self.door_scale_param = 0.0

        self.action_scale =  0.2 # left 0.2 # right_pull 0.4
        self.start_pos_noise_scale = 0 # 0.5
        self.start_rot_noise_scale = 0  # 0.25

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # reward parameters
        self.open_reward_scale = 100.0
        self.handle_reward_scale = 50.0
        self.dist_reward_scale = 1.0
        self.action_penalty_scale = 0.001

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        # self.distX_offset = 0.04 # 0.04 default
        
        self.dt = self.cfg['sim']['dt']

        # set camera properties for realsense now : 435 and 405
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 64
        self.camera_props.height = 48
        self.depth_min = -3.0
        self.depth_max = -0.07

        self.camera_props.enable_tensors = True # If False, d_img process doesnt work  

        # set observation space and action space
        self.cfg["env"]["numObservations"] = 12 + self.camera_props.width*self.camera_props.height
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        # for experimental data
        self.hinge_max_in_episode = torch.zeros(self.num_envs)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # self.franka_default_dof_pos = to_torch([0, -1.2, 0, -2.2, 0, 2.7, 0], device=self.device) # left_best
        self.franka_default_dof_pos = to_torch([0, -1.2, 0, -2.2, 0, 2.7, 0], device=self.device) # right_best

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # (num_envs*num_actors, 8, 2)

        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]  # (num_envs, 6, 2)
        # print(self.franka_dof_state.shape)
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.hand_pose_euler_prev = torch.zeros(self.num_envs, 6 , device=self.device)
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:] # (num_envs, 2, 2)
        # print(self.door_dof_state.shape)
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_pos_prev = torch.zeros_like(self.door_dof_pos, device=self.device)        
        self.door_dof_vel = self.door_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        # print(self.num_bodies)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

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
        franka_asset_file = "urdf/franka_description/robots/franka_hook.urdf"
        door_1_asset_file = 'urdf/door_test/door_1_wall.urdf'
        door_2_asset_file = 'urdf/door_test/door_2_wall.urdf'
        door_1_inv_asset_file = 'urdf/door_test/door_1_inv_wall.urdf'
        door_2_inv_asset_file = 'urdf/door_test/door_2_inv_wall.urdf'

        wood_texture = self.gym.create_texture_from_file(self.sim, './assets/textures/wood_3.jpg')
        # wood_texture  = 'textures/wood_1.png'

        # load franka asset
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
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

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
        

        franka_dof_stiffness = to_torch([200, 200, 200, 200, 200, 200, 200], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_1_asset)
        self.num_door_dofs = self.gym.get_asset_dof_count(door_1_asset)

        # torque tensor for door handle
        self.handle_torque_tensor = torch.zeros([self.num_envs, self.num_franka_dofs+self.num_door_dofs], dtype=torch.float, device=self.device)
        self.handle_torque_tensor[:,8] = -5

        print('----------------------------------------------- num properties ----------------------------------------')
        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num door bodies: ", self.num_door_bodies)
        print("num door dofs: ", self.num_door_dofs)
        print('----------------------------------------------- num properties ----------------------------------------')

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []

        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # franka_dof_props['hasLimits'][i] = False
                # print(f'############### feed back ####################\n{franka_dof_props}')
            franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            franka_dof_props['lower'][i] = -10
            franka_dof_props['upper'][i] = 10
            franka_dof_props['damping'][i] = franka_dof_damping[i]
            

            franka_dof_props['effort'][i] = 1000
        print(franka_dof_props)

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        # set door dof properties
        door_dof_props = self.gym.get_asset_dof_properties(door_1_asset)
    
        # start pose
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.7, -0.3, 0) # left_pull_best 
        # franka_start_pose.p = gymapi.Vec3(0.7, 0, 0) # right_best_pull
        # franka_start_pose.p = gymapi.Vec3(0.7, 0, 0) # right_trash

        franka_start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 3.14159265)

        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_door_bodies = self.gym.get_asset_rigid_body_count(door_1_asset)
        num_door_shapes = self.gym.get_asset_rigid_shape_count(door_1_asset)

        max_agg_bodies = num_franka_bodies + num_door_bodies
        max_agg_shapes = num_franka_shapes + num_door_shapes

        # camera pose setting
        camera_tf = gymapi.Transform()
        camera_tf.p = gymapi.Vec3(0.1, 0, 0.05)
        camera_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(0))

        self.camera_props.enable_tensors = True # when Vram larger

        print('#############################################################################################################')
        print(f'num_franka_bodies : {num_franka_bodies}, num_franka_shapes : {num_franka_shapes}, \nnum_door_bodies : {num_door_bodies}, num_door_shapes : {num_door_shapes}')
        print('#############################################################################################################')

        self.frankas = []
        self.doors = []
        self.envs = []
        self.camera_handles = []
        
        door_asset_count = 0
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot hand actor name as "robot_hand"
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
                                                                                                # ↑self collision ON
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            self.gym.set_rigid_body_color(env_ptr, franka_actor, 12, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))


            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # # create door actors # all doors ------------------------------------------
            # if door_asset_count == 3:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[door_asset_count], door_start_pose, "door", i, 0, 0)
            #     door_asset_count = 0
            # else:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[door_asset_count], door_start_pose, "door", i, 0, 0)
            #     door_asset_count += 1
            # # -------------------------------------------------------------------------
                
            # # only left hinge ---------------------------------------------------------
            # if i % 2 == 0:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[3], door_start_pose, "door", i, 0, 0)
            # else:
            #     door_actor = self.gym.create_actor(env_ptr, door_assets[1], door_start_pose, "door", i, 0, 0)
            # # -------------------------------------------------------------------------
                
            # only rihgt hinge ---------------------------------------------------------
            if i % 2 == 0:
                door_actor = self.gym.create_actor(env_ptr, door_assets[0], door_start_pose, "door", i, 0, 0)
            else:
                door_actor = self.gym.create_actor(env_ptr, door_assets[2], door_start_pose, "door", i, 0, 0)
            # -------------------------------------------------------------------------
                
            self.gym.set_actor_dof_properties(env_ptr, door_actor, door_dof_props)
            #door size randomization ###########################################################################
            # self.gym.set_actor_scale(env_ptr, door_actor, 1.0 + (torch.rand(1) - 0.5) * self.door_scale_param)
            ####################################################################################################
            self.gym.set_actor_scale(env_ptr, door_actor, 0.9)


            # door color
            # self.gym.set_rigid_body_color(env_ptr, door_actor, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))
            # door texture
            # self.gym.set_rigid_body_texture(env_ptr, door_actor, 1, gymapi.MESH_VISUAL, wood_texture)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.doors.append(door_actor)

            camera_handle = self.gym.create_camera_sensor(self.envs[i], self.camera_props)
            self.camera_handles.append(camera_handle)
            camera_mnt = self.gym.find_actor_rigid_body_handle(self.envs[i], franka_actor, "ee_rz_link")
            self.gym.attach_camera_to_body(camera_handle, self.envs[i], camera_mnt, camera_tf, gymapi.FOLLOW_TRANSFORM)

            # l_color = gymapi.Vec3(1,1,1)
            # l_ambient = gymapi.Vec3(1,1,1)
            # l_direction = gymapi.Vec3(1,1,1)
            # self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

        # handles definition : index
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_handle")
        # print(self.hand_handle)
        # self.hook_pose = self.dof_state
        self.door_handle = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "door_handles")
        # print('------------self.door_handle',self.door_handle)
        self.init_data()

    def init_data(self): # NOT SURE NEED
        #get rigid states
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        # franka information
        self.hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_handle")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], self.hand) # robot 座標系からの pose (0, 0, 0.5, Quat(0,0,1,0))


        #moriya change


        self.ee_states = self.rigid_body_states[:, self.hand_handle][:, :7]
        self.ee_pose = self.ee_states[:, :3]
        self.ee_quat = self.ee_states[:, 3:]
        # self.ee_eular = torch.stack([
        #     self.quat_to_eular(self.ee_quat[env, :])
        #     for env in range(self.num_envs)], dim = 0).to(self.device)
        # self.ee_eular = torch.tensor([self.ee_eular[0], self.ee_eular[1], self.ee_eular[2]])
        # print(len(self.ee_eular))

        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, 'franka')
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        # self.jacobian_end_effector = self.jacobian[:, self.gym.find_actor_rigid_body_handle(self.envs[i], , "ee_rz_link")]
        self.jacobian_end_effector = self.jacobian[:, self.hand_handle-1, :, :]

                
        
    def compute_reward(self, actions): #if you edit, go to jitscripts

        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.door_dof_pos, self.door_dof_pos_prev, self.hand_dist, 
            self.num_envs, 
            self.open_reward_scale, self.handle_reward_scale, self.dist_reward_scale, self.action_penalty_scale, self.max_episode_length)
        

    def debug_camera_imgs(self):
        
        import cv2
        for j in range(self.num_envs):
            # d_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_DEPTH)
            # np.savetxt(f"./.test_data/d_img_{j}.csv",d_img, delimiter=',')
            rgb_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_COLOR)
            rgb_img = rgb_img.reshape(rgb_img.shape[0],-1,4)[...,:3]
            cv2.imshow(f'rgb{j}', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            torch.save(self.pp_d_imgs[0, :], f'./.test_data/pp_.d_img')
            torch.save(self.th_n_d_imgs[0, :], f'./.test_data/th_n_.d_img')


    def d_img_process(self):

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        d_imgs = torch.stack([
            gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)).view(self.camera_props.height * self.camera_props.width)
            for env, camera_handle in zip(self.envs, self.camera_handles)]).to(self.device)
        # print(torch.max(d_imgs), torch.min(d_imgs))

        thresh_d_imgs = torch.where(torch.logical_or(d_imgs <= self.depth_min, d_imgs >= self.depth_max), 0, d_imgs)
        # print('thresh_raw', torch.max(thresh_d_imgs), torch.min(thresh_d_imgs))

        self.th_n_d_imgs = (thresh_d_imgs - self.depth_min)/(self.depth_max - self.depth_min)
        # print('thresh_norm', torch.max(self.th_n_d_imgs), torch.min(self.th_n_d_imgs))
        # print('thresh_norm all', self.th_n_d_imgs)

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

        # franka rigid body states
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3] # hand position
        # print(hand_pos.shape)
        hand_rot_quat = self.rigid_body_states[:, self.hand_handle][:, 3:7] # hand orientation --quat
        hand_rot_euler = self.quat_to_eular(hand_rot_quat[0,...]).view(1, -1).to(self.device)
        # print(hand_rot_euler.shape)
        self.hand_pose_euler = torch.cat([hand_pos, hand_rot_euler], dim=-1)
        # print('hand_pose_euler_prev', self.hand_pose_euler_prev.shape)
        # print('hand_pose_euler',self.hand_pose_euler.shape)

        # hand_pose = self.rigid_body_states[:, self.hand_handle][:, 0:7] # hand_pose
        # print(hand_pose.shape)
        hand_vel = self.rigid_body_states[:, self.hand_handle][:, 7:13] # hand lin_euler_vel
        #  = self.rigid_body_states[:, self.hand_handle][:, 10:13] # hand ang_vel --euler

        # door handle rigid body states
        door_handle_pos = self.rigid_body_states[:, self.door_handle][:, 0:3]
        self.hand_dist = torch.norm(door_handle_pos - hand_pos, dim = 1)

        dof_pos_dt = self.hand_pose_euler - self.hand_pose_euler_prev
        # print(dof_pos_dt)
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:] # (num_envs, 2, 2)
        self.door_dof_pos = self.door_dof_state[..., 0] # shape : (num_envs, 2)
        
        # self.door_dof_vel = self.door_dof_state[..., 1] 

        # self.obs_buf = torch.cat((dof_pos_dt, self.franka_dof_vel, self.dist_d_imgs), dim = -1)

        self.obs_buf = torch.cat((dof_pos_dt, hand_vel, self.pp_d_imgs), dim = -1)
        # print(self.dist_d_imgs)
        # print('observation space size:', self.obs_buf.shape)

        return self.obs_buf    
        
    def reset_idx(self, env_ids):
        # env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka ： tensor_clamp from torch_jit utils action dimension limitations
        # -0.25 - 0.25 noise

        # with no limit
        rand_pos = -1 * torch.rand(len(env_ids), 3, device=self.device) 
        rand_rot = -1 * torch.rand(len(env_ids), 3, device=self.device)
        rand_pos += 0.5
        rand_rot += 0.5
        # rand_rot[:,1] *= 0.5 # smallen pitch
        rand_pos = rand_pos * self.start_pos_noise_scale
        rand_rot = rand_rot * self.start_rot_noise_scale

        # print(rand_pos)
        # pos = self.franka_default_dof_pos.unsqueeze(0) + 0.75 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5)
        # pos = self.franka_default_dof_pos.unsqueeze(0) + torch.cat([rand_pos , rand_rot], dim=-1)
        # print(pos)
        # with limit
        # pos = tensor_clamp(
        #     self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
        #     self.franka_dof_lower_limits, self.franka_dof_upper_limits)            

        self.franka_dof_pos[env_ids, :] = self.franka_default_dof_pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = self.franka_default_dof_pos

        # reset door dof state
        self.door_dof_state[env_ids, :] = torch.zeros_like(self.door_dof_state[env_ids])
        self.door_dof_pos_prev[env_ids, :] = torch.zeros_like(self.door_dof_pos_prev[env_ids])       
        self.hand_pose_euler_prev[env_ids, :] = torch.zeros_like(self.hand_pose_euler_prev[env_ids])       

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        
    def pre_physics_step(self, actions): # self.gym.set_dof_target_tensor()
        self.actions = actions.clone().to(self.device)
        self.actions[:,0]*=-1
        self.actions[:,1]*=-1
        # self.actions[:,5]*=-1
        # self.actions[:,2]*=-1
        self.actions = self.dt * self.actions * self.action_scale
        # print('self.actions',self.actions)
        # self.actions = self.zero_actions()
        # self.actions = self.uni_actions()
        # print('self.actions', self.actions) # for debug

        # current_hand = self.rigid_body_states[;, self.hand]

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        jacobian = self.jacobian_end_effector

        current_position = self.ee_pose
        # current_rpy = self.ee_eular
        current_quat = self.ee_quat

        self.current_rpy = torch.stack([
            self.quat_to_eular(self.ee_quat[env, :])
            for env in range(self.num_envs)], dim = 0).to(self.device)


        # targets = self.franka_dof_targets[:, :self.num_franka_dofs] +   self.dt * self.actions * self.action_scale
        # print(self.actions.device)
        # print(self.ee_pose.device)
        goal_position = self.ee_pose + self.actions[:, :3]
        goal_rpy = self.current_rpy + self.actions[:, 3:]
        goal_quat = torch.stack([
            self.eular_to_quat(goal_rpy[env, :])
            for env in range(self.num_envs)], dim = 0).to(self.device)
        # goal_orientation = torch.tensor([goal_quat[:, ], goal_quat[1], goal_quat[2], goal_quat[3]])

        d_theta = ik(jacobian, current_position, current_quat, goal_position, goal_quat, 0.01)
        print(d_theta)

        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + d_theta
        # print(targets)




        # -----------with clamp limit --------------------------------------
        # self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
        #     targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # ------------------------------------------------------------------
        # -----------without clamp limit------------------------------------
        self.franka_dof_targets[:, :self.num_franka_dofs] = targets 
        # ------------------------------------------------------------------
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets))    
        # self.gym.set_rigid_body_state()

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        # self.compute_data()
        self.door_dof_pos_prev = self.door_dof_pos.clone()

        print(self.hand_pose_euler.shape)
        self.hand_pose_euler_prev = self.hand_pose_euler.clone()
        
        # print('prev_pos:',self.franka_dof_pos_prev)


    def compute_data(self):
        
        for env in range(self.num_envs):
            if self.door_dof_pos[env, 0] > self.hinge_max_in_episode[env]:
                self.hinge_max_in_episode[env] = self.door_dof_pos[env, 0]
            else:
                pass
        # print(self.hinge_max_in_episode)

        if self.progress_buf[0] == (self.max_episode_length-1):
        # if self.progress_buf[0] == 10:
            print(self.hinge_max_in_episode.shape)
            import time
            with open('./.test_data/door_hinges.csv', 'ab') as fd:
                
                np.savetxt(fd, 57.296*self.hinge_max_in_episode.view(1,-1).detach().cpu().numpy(), delimiter=',')
            self.hinge_max_in_episode = torch.zeros(self.num_envs)
            self.data_count += 1
        else:
            pass

    
    def quat_to_eular(self, quat):
        quat = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
        eular = quat.to_euler_zyx()

        eular_tensor = torch.tensor([eular[0], eular[1], eular[2]])

        eular_tensor = torch.nan_to_num(eular_tensor, nan=0)
        return eular_tensor

    def eular_to_quat(self, eular):
        quat = gymapi.Quat.from_euler_zyx(eular[0], eular[1], eular[2])
        # quat = gymtorch.wrap_tensor(quat)

        quat_tensor  = torch.tensor([quat.x, quat.y, quat.z, quat.w])

        return quat_tensor

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, door_dof_pos, door_dof_pos_prev, hand_dist, num_envs, open_reward_scale, handle_reward_scale, dist_reward_scale,
    action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    # print(open_reward_scale, handle_reward_scale, dist_reward_scale, action_penalty_scale)
    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(-1*actions ** 2, dim=-1) * action_penalty_scale

    open_reward = door_dof_pos[:,0] * open_reward_scale    # additional reward to open
    handle_reward = door_dof_pos[:,1] * handle_reward_scale
    # print(hand_dist)
    hand_dist_thresh = torch.where(hand_dist < 0.15, torch.zeros_like(hand_dist), hand_dist)

    # dist_reward = -1 * hand_dist * dist_reward_scale
    dist_reward = -1 * hand_dist_thresh * dist_reward_scale
    dist_reward_no_thresh = -1 * (hand_dist + torch.log(hand_dist + 0.005)) * dist_reward_scale

    rewards = open_reward + dist_reward + handle_reward + action_penalty

    # # print(hand_dist)
    # print('----------------open_reward max:',torch.max(open_reward))
    # print('--------------handle_reward max:', torch.max(handle_reward))
    # print('----------------dist_min:', torch.min(hand_dist))
    # print('-------------action_penalty max:', torch.min(action_penalty))

    # rewards = open_reward + dist_reward_no_thresh + handle_reward + action_penalty
    # rewards = dist_reward_no_thresh + action_penalty

    # success reward
    # rewards = torch.where(door_dof_pos[:,0] > 1.55, rewards + 1000, rewards)

    # # rewards = dist_reward
    # print('-------------------door_hinge_max :', torch.max(door_dof_pos[:,0]), 'door_hinge_min :', torch.min(door_dof_pos[:,0]))
    # print('-------------------door_handle_max :', torch.max(door_dof_pos[:,1]), 'door_handle_min :', torch.min(door_dof_pos[:,1]))
    # print('----------------------rewards_max :', torch.max(rewards), 'rewards_min :',torch.min(rewards))

    # reset_buf = torch.where(door_dof_pos[:, 0] >= 1.56, torch.ones_like(reset_buf), reset_buf)


    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

@torch.jit.script
def compute_d_imgs(d_imgs, depth_min, depth_max, replace_val): # wasnt better than normal function

    # type: (Tensor, float, float, float) -> Tensor 
    condition = torch.logical_or(d_imgs < depth_min, d_imgs > depth_max)
    dist_d_imgs = (d_imgs - depth_min)/(depth_max - depth_min)
    dist_d_imgs = torch.where(condition, replace_val, dist_d_imgs)
    return dist_d_imgs 












        


