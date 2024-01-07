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
# from base.vec_task import VecTask

import torch


class DoorHook(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.n = 0
        self.max_episode_length = 300

        self.action_scale = 1.5
        self.start_position_noise = 1.0
        self.start_rotation_noise = 0.2
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # reward parameters
        self.open_reward_scale = 1000.0
        self.handle_reward_scale = 200.0
        self.dist_reward_scale = 10.0
        self.action_penalty_scale = 0.005

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

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.ur3_default_dof_pos = to_torch([0, 0, 0, 0, 0, 0], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # (num_envs*num_actors, 8, 2)

        self.ur3_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur3_dofs]  # (num_envs, 6, 2)
        # print(self.ur3_dof_state.shape)
        self.ur3_dof_pos = self.ur3_dof_state[..., 0]
        self.ur3_dof_pos_prev = torch.zeros_like(self.ur3_dof_pos, device=self.device)
        self.ur3_dof_vel = self.ur3_dof_state[..., 1]
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur3_dofs:] # (num_envs, 2, 2)
        # print(self.door_dof_state.shape)
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_pos_prev = torch.zeros_like(self.door_dof_pos, device=self.device)        
        self.door_dof_vel = self.door_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        # print(self.num_bodies)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur3_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

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
        ur3_asset_file = "urdf/door_test/hook_test.urdf"
        door_asset_file = 'urdf/door_test/door_1.urdf'
        
        # load ur3 asset
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
        ur3_asset = self.gym.load_asset(self.sim, asset_root, ur3_asset_file, asset_options)

        # load door asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        door_asset = self.gym.load_asset(self.sim, asset_root, door_asset_file, asset_options)

        ur3_dof_stiffness = to_torch([500, 500, 500, 500, 500, 500], dtype=torch.float, device=self.device)
        ur3_dof_damping = to_torch([10, 10, 10, 10, 10, 10], dtype=torch.float, device=self.device)

        self.num_ur3_bodies = self.gym.get_asset_rigid_body_count(ur3_asset)
        self.num_ur3_dofs = self.gym.get_asset_dof_count(ur3_asset)
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        self.num_door_dofs = self.gym.get_asset_dof_count(door_asset)

        # torque tensor for door handle
        self.handle_torque_tensor = torch.zeros([self.num_envs, self.num_ur3_dofs+self.num_door_dofs], dtype=torch.float, device=self.device)
        self.handle_torque_tensor[:,7] = -10

        # print('----------------------------------------------- num properties ----------------------------------------')
        # print("num ur3 bodies: ", self.num_ur3_bodies)
        # print("num ur3 dofs: ", self.num_ur3_dofs)
        # print("num door bodies: ", self.num_door_bodies)
        # print("num door dofs: ", self.num_door_dofs)
        # print('----------------------------------------------- num properties ----------------------------------------')

        # set ur3 dof properties
        ur3_dof_props = self.gym.get_asset_dof_properties(ur3_asset)
        self.ur3_dof_lower_limits = []
        self.ur3_dof_upper_limits = []

        for i in range(self.num_ur3_dofs):
            ur3_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # ur3_dof_props['hasLimits'][i] = False
                # print(f'############### feed back ####################\n{ur3_dof_props}')
            ur3_dof_props['stiffness'][i] = ur3_dof_stiffness[i]
            ur3_dof_props['lower'][i] = -10
            ur3_dof_props['upper'][i] = 10

            ur3_dof_props['effort'][i] = 500
        print(ur3_dof_props)

        self.ur3_dof_lower_limits = to_torch(self.ur3_dof_lower_limits, device=self.device)
        self.ur3_dof_upper_limits = to_torch(self.ur3_dof_upper_limits, device=self.device)
        self.ur3_dof_speed_scales = torch.ones_like(self.ur3_dof_lower_limits)

        # set door dof properties
        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
    
        # start pose
        ur3_start_pose = gymapi.Transform()
        ur3_start_pose.p = gymapi.Vec3(0.5, -0.3, 1.02) # initial position of the robot
        ur3_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        # compute aggregate size
        num_ur3_bodies = self.gym.get_asset_rigid_body_count(ur3_asset)
        num_ur3_shapes = self.gym.get_asset_rigid_shape_count(ur3_asset)
        num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        num_door_shapes = self.gym.get_asset_rigid_shape_count(door_asset)

        max_agg_bodies = num_ur3_bodies + num_door_bodies
        max_agg_shapes = num_ur3_shapes + num_door_shapes

        # camera pose setting
        camera_tf = gymapi.Transform()
        camera_tf.p = gymapi.Vec3(-0.065, 0, 0.131)
        camera_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(0))

        self.camera_props.enable_tensors = True # when Vram larger

        print('#############################################################################################################')
        print(f'num_ur3_bodies : {num_ur3_bodies}, num_ur3_shapes : {num_ur3_shapes}, \nnum_door_bodies : {num_door_bodies}, num_door_shapes : {num_door_shapes}')
        print('#############################################################################################################')

        self.ur3s = []
        self.doors = []
        self.envs = []
        self.camera_handles = []
        
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot hand actor name as "robot_hand"
            ur3_actor = self.gym.create_actor(env_ptr, ur3_asset, ur3_start_pose, "robot_hand", i, 0, 0)
                                                                                                # ↑self collision ON
            self.gym.set_actor_dof_properties(env_ptr, ur3_actor, ur3_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            door_pose = door_start_pose
            # dx = np.random.rand() - 0.5
            # door_pose.p.x = self.start_position_noise * dx
            # dz = np.random.rand() - 0.5
            # dy = np.random.rand() - 0.5
            # door_pose.p.y = self.start_position_noise * dy
            # door_pose.r.z = self.start_rotation_noise * dz
            # door_pose.p.z += self.start_position_noise * dz
            door_actor = self.gym.create_actor(env_ptr, door_asset, door_pose, "door", i, 0, 0) # 0 : self collision ON
            self.gym.set_actor_dof_properties(env_ptr, door_actor, door_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.ur3s.append(ur3_actor)
            self.doors.append(door_actor)

            camera_handle = self.gym.create_camera_sensor(self.envs[i], self.camera_props)
            self.camera_handles.append(camera_handle)
            camera_mnt = self.gym.find_actor_rigid_body_handle(self.envs[i], ur3_actor, "ee_rz_link")
            self.gym.attach_camera_to_body(camera_handle, self.envs[i], camera_mnt, camera_tf, gymapi.FOLLOW_TRANSFORM)

        # handles definition : index
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "hook_finger")
        # print(self.hand_handle)
        # self.hook_pose = self.dof_state
        self.door_handle = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "door_handles")
        # print('------------self.door_handle',self.door_handle)
        self.init_data()

    def init_data(self): # NOT SURE NEED
        # ur3 information
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur3s[0], "ee_rz_link")
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand) # robot 座標系からの pose (0, 0, 0.5, Quat(0,0,1,0))
                
        
    def compute_reward(self, actions): #if you edit, go to jitscripts

        self.rew_buf[:], self.reset_buf[:] = compute_ur3_reward(
            self.reset_buf, self.progress_buf, self.actions, self.door_dof_pos, self.door_dof_pos_prev, self.hand_dist, 
            self.num_envs, 
            self.open_reward_scale, self.handle_reward_scale, self.dist_reward_scale, self.action_penalty_scale, self.max_episode_length)
        

    def debug_camera_imgs(self):
        
        import cv2
        for j in [0,1,2,3,4,5]:
            # d_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_DEPTH)
            # np.savetxt(f"./.test_data/d_img_{j}.csv",d_img, delimiter=',')
            rgb_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_COLOR)
            rgb_img = rgb_img.reshape(rgb_img.shape[0],-1,4)[...,:3]
            cv2.imshow(f'rgb{j}', rgb_img)
            cv2.waitKey(1)

    def d_img_process(self):

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.d_imgs = torch.stack([
            gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)).view(self.camera_props.height * self.camera_props.width)
            for env, camera_handle in zip(self.envs, self.camera_handles)]).to(self.device)
        
        norm_d_imgs = (self.d_imgs - self.depth_min)/(self.depth_max - self.depth_min)
        norm_d_imgs[torch.where(torch.logical_or(self.d_imgs < self.depth_min, self.d_imgs > self.depth_max))] = -1.0
        self.pp_d_imgs = norm_d_imgs

        # self.get_d_img_dataset()

        self.gym.end_access_image_tensors(self.sim)
        
        # print(self.pp_d_imgs[0]) # debug

    def get_d_img_dataset(self):

        for z in range(self.num_envs):
            torch.save(self.pp_d_imgs[z, :], f'../../depthnet/depth_dataset/new_{self.n}_{z}.d_img')
        self.n = self.n + 1

    def compute_observations(self):  # NOW DEFINING
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.d_img_process()
        # self.debug_camera_imgs()

        #apply door handle torque_tensor as spring actuation
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.handle_torque_tensor))

        # ur3 rigid body states
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3] # hand position
        # hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7] # hand orientation
        # hand_vel_pos = self.rigid_body_states[:, self.hand_handle][:, 7:10] # hand lin_vel
        # hand_vel_rot = self.rigid_body_states[:, self.hand_handle][:, 10:13] # hand ang_vel

        # door handle rigid body states
        door_handle_pos = self.rigid_body_states[:, self.door_handle][:, 0:3]
        # print(hand_pos[0])
        # print(door_handle_pos[0])

        self.hand_dist = torch.norm(door_handle_pos - hand_pos, dim = 1)
        # print(self.hand_dist.shape)

        # print(self.door_handle) index of door_handle
        # door_rot = self.rigid_body_states[:, self.door_handle][:, 3:7]
        # door_vel_pos = self.rigid_body_states[:, self.door_handle][:, 7:10]
        # door_vel_rot = self.rigid_body_states[:, self.door_handle][:, 10:13]

        # ur3 dof states [x y z rx ry rz] to obs_buf
        # self.ur3_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur3_dofs] # (num_envs, 6, 2)
        # self.ur3_dof_pos = self.ur3_dof_state[...,0]
        dof_pos_dt = self.ur3_dof_pos - self.ur3_dof_pos_prev
        
        # print('prev_pos:', self.ur3_dof_pos_prev)
        # print('current_pos:',self.ur3_dof_pos)
        # print('dt:', dof_pos_dt)
        # print('vel',self.ur3_dof_vel)

        # print('prev_pos:',self.ur3_dof_pos_prev)
        # print('current_pos:', self.ur3_dof_pos)
        # self.ur3_dof_vel = self.ur3_dof_state[...,1]

        # door dof states [hinge handle] ang to obs_buf
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur3_dofs:] # (num_envs, 2, 2)
    
        self.door_dof_pos = self.door_dof_state[..., 0] # shape : (num_envs, 2)
        
        # self.door_dof_vel = self.door_dof_state[..., 1] 

        # door_hinge_ang = self.door_dof_pos[:,0] # 0 : hinge, 1 : handle
        # # print('doof_hinge_ang :', door_hinge_ang[0])
        # door_hinge_vel = self.door_dof_vel[:,0]
        # door_handle_ang = self.door_dof_pos[:,1]
        # door_handle_vel = self.door_dof_vel[:,1]
        # define obsefcation space
        self.obs_buf = torch.cat((dof_pos_dt, self.ur3_dof_vel, self.pp_d_imgs), dim = -1)
        # print(self.pp_d_imgs)
        # print('observation space size:', self.obs_buf.shape)

        return self.obs_buf    
        
    def reset_idx(self, env_ids):
        # env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset ur3 ： tensor_clamp from torch_jit utils action dimension limitations
        # -0.25 - 0.25 noise
        # with no limit
        rand_pos = -1 * torch.rand(len(env_ids), self.num_ur3_dofs, device=self.device)
        rand_pos[:,1:] += 0.5
        # print(rand_pos)
        # pos = self.ur3_default_dof_pos.unsqueeze(0) + 0.75 * (torch.rand((len(env_ids), self.num_ur3_dofs), device=self.device) - 0.5)
        pos = self.ur3_default_dof_pos.unsqueeze(0) + 1.0 * rand_pos
        # print(pos)
        # with limit
        # pos = tensor_clamp(
        #     self.ur3_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_ur3_dofs), device=self.device) - 0.5),
        #     self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)            

        self.ur3_dof_pos[env_ids, :] = pos
        self.ur3_dof_vel[env_ids, :] = torch.zeros_like(self.ur3_dof_vel[env_ids])
        self.ur3_dof_targets[env_ids, :self.num_ur3_dofs] = pos

        # reset door dof state
        self.door_dof_state[env_ids, :] = torch.zeros_like(self.door_dof_state[env_ids])
        self.door_dof_pos_prev[env_ids, :] = torch.zeros_like(self.door_dof_pos_prev[env_ids])       
        self.ur3_dof_pos_prev[env_ids, :] = torch.zeros_like(self.ur3_dof_pos_prev[env_ids])       

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur3_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        
    def pre_physics_step(self, actions): # self.gym.set_dof_target_tensor()
        self.actions = actions.clone().to(self.device)
        # print('self.actions',self.actions)
        # self.actions = self.zero_actions()
        # self.actions = self.uni_actions()
        # print('self.actions', self.actions) # for debug
        targets = self.ur3_dof_targets[:, :self.num_ur3_dofs] +   self.dt * self.actions * self.action_scale
        # -----------with clamp limit --------------------------------------
        # self.ur3_dof_targets[:, :self.num_ur3_dofs] = tensor_clamp(
        #     targets, self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)
        # ------------------------------------------------------------------
        # -----------without clamp limit------------------------------------
        self.ur3_dof_targets[:, :self.num_ur3_dofs] = targets 
        # ------------------------------------------------------------------
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.ur3_dof_targets))    

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.door_dof_pos_prev = self.door_dof_pos.clone()
        self.ur3_dof_pos_prev = self.ur3_dof_pos.clone()
        
        # print('prev_pos:',self.ur3_dof_pos_prev)


        # debug viz
        # if self.debug_viz:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)

        #     for i in range(self.num_envs):
        #         px = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.ur3_grasp_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

        #         px = (self.door_grasp_pos[i] + quat_apply(self.door_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.door_grasp_pos[i] + quat_apply(self.door_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.door_grasp_pos[i] + quat_apply(self.door_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.door_grasp_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

        #         px = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.ur3_lfinger_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

        #         px = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.ur3_rfinger_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ur3_reward(
    reset_buf, progress_buf, actions, door_dof_pos, door_dof_pos_prev, hand_dist, num_envs, open_reward_scale, handle_reward_scale, dist_reward_scale,
    action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    # print(open_reward_scale, handle_reward_scale, dist_reward_scale, action_penalty_scale)
    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(-1*actions ** 2, dim=-1) * action_penalty_scale
    # handle_reward=torch.zeros(1,num_envs)
    # open_reward = door_dof_pos[:,0] * door_dof_pos[:,0]
    # open_reward = (door_dof_pos[:,0] - door_dof_pos_prev[:,0]) * open_reward_scale
    open_reward = ((door_dof_pos[:,0] * door_dof_pos[:,0]) + (door_dof_pos[:,0] - door_dof_pos_prev[:,0])) * open_reward_scale
    # additional reward to open
    handle_reward = (door_dof_pos[:,1] * door_dof_pos[:,1]) * handle_reward_scale
    # print(hand_dist)
    hand_dist_thresh = torch.where(hand_dist < 0.20, torch.zeros_like(hand_dist), hand_dist)

    # dist_reward = -1 * hand_dist * dist_reward_scale
    dist_reward = -1 * hand_dist_thresh * dist_reward_scale



    # print(hand_dist)
    print('----------------open_reward max:',torch.max(open_reward))
    print('--------------handle_reward max:', torch.max(handle_reward))
    print('----------------dist_reward max:', torch.max(dist_reward))
    print('-------------action_penalty max:', torch.min(action_penalty))

    # edited reward to diff_hinge_ang handle_rew.

    # action penalty must be minus??
    rewards = open_reward + dist_reward + handle_reward + action_penalty
    # rewards = dist_reward + action_penalty

    # success reward
    rewards = torch.where(door_dof_pos[:,0] > 1.55, rewards + 1000, rewards)

    # rewards = dist_reward
    print('-------------------door_hinge_max :', torch.max(door_dof_pos[:,0]), 'door_hinge_min :', torch.min(door_dof_pos[:,0]))
    print('-------------------door_handle_max :', torch.max(door_dof_pos[:,1]), 'door_handle_min :', torch.min(door_dof_pos[:,1]))
    print('----------------------rewards_max :', torch.max(rewards), 'rewards_min :',torch.min(rewards))

    reset_buf = torch.where(door_dof_pos[:, 0] >= 1.56, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

@torch.jit.script
def compute_d_imgs(d_imgs, depth_min, depth_max, replace_val): # wasnt better than normal function

    # type: (Tensor, float, float, float) -> Tensor 
    condition = torch.logical_or(d_imgs < depth_min, d_imgs > depth_max)
    norm_d_imgs = (d_imgs - depth_min)/(depth_max - depth_min)
    norm_d_imgs = torch.where(condition, replace_val, norm_d_imgs)
    return norm_d_imgs 












        


