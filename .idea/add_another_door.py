"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Apply Forces (apply_forces.py)
----------------------------
This example shows how to apply forces and torques to rigid bodies using the tensor API.
"""

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch
import time

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Example of applying forces and torques to bodies")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# load ball asset
asset_root = "../assets"
door_1 = "urdf/door_test/door_1_wall.urdf"
door_2 = 'urdf/door_test/door_2_wall.urdf'
door_1_inv = 'urdf/door_test/door_1_inv_wall.urdf'
door_2_inv = 'urdf/door_test/door_2_inv_wall.urdf'

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset_options.armature = 0.005


asset_1 = gym.load_asset(sim, asset_root, door_1, asset_options)
asset_2 = gym.load_asset(sim, asset_root, door_2, asset_options)
asset_1_inv = gym.load_asset(sim, asset_root, door_1_inv, asset_options)
asset_2_inv = gym.load_asset(sim, asset_root, door_2_inv, asset_options)

assets = [asset_1, asset_2, asset_1_inv, asset_2_inv]

door_dof_props = gym.get_asset_dof_properties(asset_1)

num_bodies = gym.get_asset_rigid_body_count(asset_1)
num_dofs = gym.get_asset_dof_count(asset_1)
print('num_bodies', num_bodies)
print('num_dofs', num_dofs)

# default pose
pose = gymapi.Transform()
pose.p.z = 0.0

# set up the env grid
num_envs = 4
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# set random seed
np.random.seed(17)

envs = []
handles = []
assets_load_count = 0
for i in range(num_envs):
    
    asset = assets[assets_load_count]
    if assets_load_count == 3:
        assets_load_count = 0
    else:
        assets_load_count += 1
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])
    
    ahandle = gym.create_actor(env, asset, pose, "door", i, 0, 0)
    gym.set_actor_dof_properties(env, ahandle, door_dof_props)

    handles.append(ahandle)
    # gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 20, 5), gymapi.Vec3(0, 0, 1))

gym.prepare_sim(sim)

door_handle_idx = gym.get_actor_dof_handle(envs[0], ahandle, 1)

torque_amt = 0

frame_count = 0
while not gym.query_viewer_has_closed(viewer):

    if (frame_count - 99) % 200 == 0:
        # set forces and torques for the ant root bodies
        torques = torch.zeros((num_envs, 2), device=device, dtype=torch.float)
        indexes = torch.zeros((num_envs), device=device, dtype=torch.int)
        indexes = torch.tensor([0,1,2,3], device=device, dtype=torch.int)
        # indexes = gymapi.DOMAIN_ACTOR
        print('indexex',indexes)
        # forces[:, 0, 2] = 300
        torques[:,1] = 0 # 50
        torques[:,0] = 0 # 100
        print(torques)
        # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
        gym.set_dof_actuation_force_tensor_indexed(sim, gymtorch.unwrap_tensor(torques) ,gymtorch.unwrap_tensor(indexes) , num_envs)
        torque_amt = -torque_amt
    elif (frame_count - 99) % 100 == 0:
        # set forces and torques for the ant root bodies
        torques = torch.zeros((num_envs, 2), device=device, dtype=torch.float)
        indexes = torch.zeros((num_envs), device=device, dtype=torch.int)
        indexes = torch.tensor([0,1,2,3], device=device, dtype=torch.int)
        print('indexes shape',indexes.shape)
        # forces[:, 0, 2] = 300
        torques[:,0] = 0 # -50
        # time.sleep(1)
        # torques[:,1] = -50
        print(torques)
        # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
        gym.set_dof_actuation_force_tensor_indexed(sim, gymtorch.unwrap_tensor(torques) , gymtorch.unwrap_tensor(indexes), num_envs)
        torque_amt = -torque_amt

    elif (frame_count - 99) % 170 == 0:
        # set forces and torques for the ant root bodies
        torques = torch.zeros((num_envs, 2), device=device, dtype=torch.float)
        indexes = torch.zeros((num_envs), device=device, dtype=torch.int)
        indexes = torch.tensor([0,1,2,3], device=device, dtype=torch.int)
        print('indexes shape',indexes.shape)
        # forces[:, 0, 2] = 300
        torques[:,0] =  0 #-50
        # time.sleep(1)
        torques[:,1] = 0  # -50
        print(torques)
        # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
        gym.set_dof_actuation_force_tensor_indexed(sim, gymtorch.unwrap_tensor(torques) , gymtorch.unwrap_tensor(indexes), num_envs)
        torque_amt = -torque_amt


    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1
    # print(frame_count)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
