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

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04 # 0.04 default
        self.dt = cfg["sim"]["dt"]

        self.cfg["env"]["numObservations"] = 17
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
        print(self.dof_state.shape) 
        test = self.dof_state.view(self.num_envs, -1 ,2)
        print(test.shape) # 8になってるから，ur3とドアのdof_stateが一緒に格納されている
        self.ur3_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur3_dofs]  # (num_envs, 6, 2)
        print(self.ur3_dof_state.shape)
        self.ur3_dof_pos = self.ur3_dof_state[..., 0]
        self.ur3_dof_vel = self.ur3_dof_state[..., 1]
        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur3_dofs:] # (num_envs, 2, 2)
        print(self.door_dof_state.shape)
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_vel = self.door_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        print(self.num_bodies)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur3_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row): # 環境をspacingおきに生成するための関数
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        ur3_asset_file = "urdf/door_test/hook_test.urdf"
        door_asset_file = 'urdf/door_test/door_1.urdf'

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            ur3_asset_file = self.cfg["env"]["asset"].get("assetFileNameUr3", ur3_asset_file)
            door_asset_file = self.cfg["env"]["asset"].get("assetFileNameDoor", door_asset_file)
        
        # load ur3 asset
        asset_options = gymapi.AssetOptions()
        vh_options = gymapi.VhacdParams()

        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.vhacd_enabled = True # if True, accurate collision enabled
        vh_options.max_convex_hulls = 1000
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
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

        ur3_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        ur3_dof_damping = to_torch([80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        self.num_ur3_bodies = self.gym.get_asset_rigid_body_count(ur3_asset)
        self.num_ur3_dofs = self.gym.get_asset_dof_count(ur3_asset)
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        self.num_door_dofs = self.gym.get_asset_dof_count(door_asset)
        print('----------------------------------------------- num properties ----------------------------------------')
        print("num ur3 bodies: ", self.num_ur3_bodies)
        print("num ur3 dofs: ", self.num_ur3_dofs)
        print("num door bodies: ", self.num_door_bodies)
        print("num door dofs: ", self.num_door_dofs)
        print('----------------------------------------------- num properties ----------------------------------------')

        # set ur3 dof properties
        ur3_dof_props = self.gym.get_asset_dof_properties(ur3_asset)
        self.ur3_dof_lower_limits = []
        self.ur3_dof_upper_limits = []

        for i in range(self.num_ur3_dofs):
            ur3_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                # print(f'############### feed back ####################\n{ur3_dof_props}')
                ur3_dof_props['stiffness'][i] = ur3_dof_stiffness[i]
                ur3_dof_props['damping'][i] = ur3_dof_damping[i]
            else:
                ur3_dof_props['stiffness'][i] = 7000.0
                ur3_dof_props['damping'][i] = 50.0

            self.ur3_dof_lower_limits.append(ur3_dof_props['lower'][i])
            self.ur3_dof_upper_limits.append(ur3_dof_props['upper'][i])

        self.ur3_dof_lower_limits = to_torch(self.ur3_dof_lower_limits, device=self.device)
        self.ur3_dof_upper_limits = to_torch(self.ur3_dof_upper_limits, device=self.device)
        self.ur3_dof_speed_scales = torch.ones_like(self.ur3_dof_lower_limits)
        # self.ur3_dof_speed_scales[[0, 1]] = 0.1 # これなんだかわかってないけど，自由度に合わせることでどうにかなる

        # ur3_dof_props['effort'][4] = 200
        ur3_dof_props['effort'][5] = 200

        # set door dof properties
        door_dof_props = self.gym.get_asset_dof_properties(door_asset)
        for i in range(self.num_door_dofs):
            door_dof_props['damping'][i] = 10.0

        # start pose
        ur3_start_pose = gymapi.Transform()
        ur3_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # urdf上の初期位置と関係
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

        # camera settings
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        camera_tf = gymapi.Transform()
        camera_tf.p = gymapi.Vec3(-0.065, 0, 0.131)
        camera_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(0))
        # camera_mnt = self.gym.find_actor_rigid_body_handle(self.envs[i], ur3_actor, "ee_rz_link")
        # self.camera_props.enable_tensors = True
        # camera_handle = self.gym.create_camera_sensor(env, self.camera_props)

        print('#############################################################################################################')
        print(f'num_ur3_bodies : {num_ur3_bodies}, num_ur3_shapes : {num_ur3_shapes}, \nnum_door_bodies : {num_door_bodies}, num_door_shapes : {num_door_shapes}')
        print('#############################################################################################################')

        self.ur3s = []
        self.doors = []
        self.envs = []
        self.camera_handles = []
        self.d_imgs =[]
        
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # create robot hand actor name as "robot_hand"
            ur3_actor = self.gym.create_actor(env_ptr, ur3_asset, ur3_start_pose, "robot_hand", i, 0, 0)
                                                                                           #↑1から0に変えたらself collision OK！
            self.gym.set_actor_dof_properties(env_ptr, ur3_actor, ur3_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            door_pose = door_start_pose
            door_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            door_pose.p.y += self.start_position_noise * dy
            door_pose.p.z += self.start_position_noise * dz
            door_actor = self.gym.create_actor(env_ptr, door_asset, door_pose, "door", i, 1, 0)
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

        # handle の定義をしている これは in range num_envs のループ外にある
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "ee_rz_link")
        self.door_handle = self.gym.find_actor_rigid_body_handle(env_ptr, door_actor, "door_handles")
        # self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "panda_leftfinger")
        # self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur3_actor, "panda_rightfinger")
        # self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)
        
        self.init_data()

    def init_data(self): # NOT SURE
        # ur3 information
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur3s[0], "ee_rz_link")
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand) # robot 座標系からの pose (0, 0, 0.5, Quat(0,0,1,0))
        hand_pose_inv = hand_pose.inverse()
        print(hand_pose_inv.p, hand_pose_inv.r) # debug

        # door information
        door_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.doors[0], "door_handles")
        # もしかしたら↑self.hand_handleと置き換えられるかもしれん
        print(door_handle)
        door_hinge = self.gym.find_actor_dof_handle(self.envs[0], self.doors[0], "door_hinge")
        print(door_hinge)
        
        
    def compute_reward(self, actions): # NOT DEFINED YET

        self.rew_buf[:], self.reset_buf[:] = compute_ur3_reward(
            self.reset_buf, self.progress_buf, self.actions, self.door_dof_pos,
            self.ur3_grasp_pos, self.door_grasp_pos, self.ur3_grasp_rot, self.door_grasp_rot,
            self.gripper_forward_axis, self.door_inward_axis, self.gripper_up_axis, self.door_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def debug_camera_imgs(self):
        
        import cv2
        for j in [0,511]:
            d_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_DEPTH)
            # -inf : -np.inf
            np.savetxt(f"./.test_data/d_img_{j}.csv",d_img, delimiter=',')
            rgb_img = self.gym.get_camera_image(self.sim, self.envs[j], self.camera_handles[j], gymapi.IMAGE_COLOR)
            rgb_img = rgb_img.reshape(rgb_img.shape[0],-1,4)[...,:3]
            cv2.imshow(f'rgb{j}', rgb_img)
            cv2.waitKey(1)

    def compute_observations(self):  # NOW DEFINING
        

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.gym.render_all_camera_sensors(self.sim)
        d_imgs = [self.gym.get_camera_image(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH) for env, camera_handle in zip(self.envs, self.camera_handles)]
        self.d_imgs = [np.where(np.isinf(d_img), 0, d_img) for d_img in d_imgs]
        
        
        # self.debug_camera_imgs()

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3] # hand position
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7] # hand orientation
        hand_vel_pos = self.rigid_body_states[:, self.hand_handle][:, 7:10] # hand lin_vel
        hand_vel_rot = self.rigid_body_states[:, self.hand_handle][:, 10:13] # hand ang_vel
        # door state dont need
        door_pos = self.rigid_body_states[:, self.door_handle][:, 0:3]
        door_rot = self.rigid_body_states[:, self.door_handle][:, 3:7]
        door_vel_pos = self.rigid_body_states[:, self.door_handle][:, 7:10]
        door_vel_rot = self.rigid_body_states[:, self.door_handle][:, 10:13]

        dof_pos_scaled = (2.0 * (self.ur3_dof_pos - self.ur3_dof_lower_limits)
                          / (self.ur3_dof_upper_limits - self.ur3_dof_lower_limits) - 1.0)
        to_target = self.door_grasp_pos - self.ur3_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.ur3_dof_vel * self.dof_vel_scale, to_target,
                                  self.door_dof_pos[:, 0].unsqueeze(-1), self.door_dof_vel[:, 0].unsqueeze(-1)), dim=-1)
                                  #                    ↑適当に3からに変更してある  ##############↑

            # zantei
        self.obs_buf = torch.cat((self.d_imgs, hand_pos, hand_rot, hand_vel_pos, hand_vel_rot))
        return self.obs_buf    
        
    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset ur3
        pos = tensor_clamp(
            self.ur3_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_ur3_dofs), device=self.device) - 0.5),
            self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)
        self.ur3_dof_pos[env_ids, :] = pos
        self.ur3_dof_vel[env_ids, :] = torch.zeros_like(self.ur3_dof_vel[env_ids])
        self.ur3_dof_targets[env_ids, :self.num_ur3_dofs] = pos

        # reset door
        self.door_dof_state[env_ids, :] = torch.zeros_like(self.door_dof_state[env_ids])

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.ur3_dof_targets[:, :self.num_ur3_dofs] + self.ur3_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.ur3_dof_targets[:, :self.num_ur3_dofs] = tensor_clamp(
            targets, self.ur3_dof_lower_limits, self.ur3_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.ur3_dof_targets))    
    
    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur3_grasp_pos[i] + quat_apply(self.ur3_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur3_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.door_grasp_pos[i] + quat_apply(self.door_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.door_grasp_pos[i] + quat_apply(self.door_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.door_grasp_pos[i] + quat_apply(self.door_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.door_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur3_lfinger_pos[i] + quat_apply(self.ur3_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur3_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur3_rfinger_pos[i] + quat_apply(self.ur3_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur3_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ur3_reward(
    reset_buf, progress_buf, actions, door_dof_pos,
    ur3_grasp_pos, door_grasp_pos, ur3_grasp_rot, door_grasp_rot,
    gripper_forward_axis, door_inward_axis, gripper_up_axis, door_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the door
    d = torch.norm(ur3_grasp_pos - door_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(ur3_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(door_grasp_rot, door_inward_axis)
    axis3 = tf_vector(ur3_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(door_grasp_rot, door_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the door (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if left finger is above the door handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(ur3_lfinger_pos[:, 2] > door_grasp_pos[:, 2],
                                       torch.where(ur3_rfinger_pos[:, 2] < door_grasp_pos[:, 2],
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the door
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(ur3_lfinger_pos[:, 2] - door_grasp_pos[:, 2])
    rfinger_dist = torch.abs(ur3_rfinger_pos[:, 2] - door_grasp_pos[:, 2])
    finger_dist_reward = torch.where(ur3_lfinger_pos[:, 2] > door_grasp_pos[:, 2],
                                     torch.where(ur3_rfinger_pos[:, 2] < door_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # how far the door has been opened out
    open_reward = door_dof_pos[:, 0] * around_handle_reward + door_dof_pos[:, 0]  # door_top_joint 0に変更してあるとりあえず

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
        + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
        + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty

    # bonus for opening door properly
    rewards = torch.where(door_dof_pos[:, 0] > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(door_dof_pos[:, 0] > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(door_dof_pos[:, 0] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # prevent bad style in opening door
    rewards = torch.where(ur3_lfinger_pos[:, 0] < door_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    rewards = torch.where(ur3_rfinger_pos[:, 0] < door_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    
    # reset if door is open or max length reached
    reset_buf = torch.where(door_dof_pos[:, 0] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos,
                             door_rot, door_pos, door_local_grasp_rot, door_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_ur3_rot, global_ur3_pos = tf_combine(
        hand_rot, hand_pos, ur3_local_grasp_rot, ur3_local_grasp_pos)
    global_door_rot, global_door_pos = tf_combine(
        door_rot, door_pos, door_local_grasp_rot, door_local_grasp_pos)

    return global_ur3_rot, global_ur3_pos, global_door_rot, global_door_pos


if __name__ == '__main__':

    DT = DoorHook()
    # DT._create_envs(10, 10, 5)











        


