"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import robosuite.utils.transform_utils as T
from robosuite.utils.control_utils import *

import torch


def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()



    n_step = 0

    # MASTER ROBOT IMPEDANCE
    Bm_v = 0.0  #30.0
    Bm_w = 0.0  #15.0
    Dm_v = 75.0  #200.0  #300.0
    Dm_w = 75.0  #200.0  #150.0
    Km_v = 150.0  #1000.0  #3000.0  #6500.0
    Km_w = 150.0  #1000.0  #2000.0  #4000.0

    # MASTER ROBOT STATES
    disp_m = np.zeros(6)
    pos_m = env.robots[0].controller.ee_pos
    ori_m = T.mat2quat(env.robots[0].controller.ee_ori_mat)
    vel_m = np.concatenate([env.robots[0].controller.ee_pos_vel, env.robots[0].controller.ee_ori_vel])
    # DESIRED STATES FROM SLAVE
    pos_md = pos_m
    ori_md = ori_m
    vel_md = vel_m

    # wave impedance
    b_v = 75.0  #200.0  #300.0
    b_w = 75.0  #200.0  #150.0
    # output wave
    u_m = np.zeros(6)
    # input wave
    v_m = np.zeros(6)

    # DESIRED FORCE TO SLAVE
    F_md = np.zeros(6)

    # TORQUE APPLIED TO MASTER
    tau_m = np.zeros(6)

    # virtual mass and inertia of the MASTER
    mass = 1.2  #10.0
    inertia = 1.2  #10.0

    #DT = env.control_timestep
    #DT2 = env.control_timestep * env.control_timestep
    DT = 0.002
    DT2 = 0.002*0.002


    # get the incoming wave v_m
    # use the initial value of v_m for the first time

    # update u_m
    u_m[0:3] = np.sqrt(2*b_v)*vel_md[0:3] - v_m[0:3]
    u_m[3:6] = np.sqrt(2*b_w)*vel_md[3:6] - v_m[3:6]

    # update vel_md
    vel_md[0:3] = (np.sqrt(2*b_v)*v_m[0:3] + F_md[0:3]) / b_v
    vel_md[3:6] = (np.sqrt(2*b_w)*v_m[3:6] + F_md[3:6]) / b_w

    # send u_m to slave
    # skip for the initial value

    # update pos_md & ori_md
    pos_md = pos_md + vel_md[0:3]*DT
    ori_md = T.quat_multiply(T.axisangle2quat(vel_md[3:6]*DT), ori_md)
    if ori_md[-1] < 0.0:
        ori_md = -ori_md

    # sample spacemouse displacement
    active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]
    action, grasp = input2action(
        device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
    )
    disp_m = action[0:6]

    # apply tau_m and update pos_m & ori_m
    pos_m = pos_m + disp_m[0:3] - 0.5*tau_m[0:3]*DT2/mass
    ori_m = T.quat_multiply(T.axisangle2quat(disp_m[3:6] - 0.5*tau_m[3:6]*DT2/inertia), ori_m)
    if ori_m[-1] < 0.0:
        ori_m = -ori_m

    # update vel_m
    vel_m[0:3] = disp_m[0:3]/DT - tau_m[0:3]*DT/mass
    vel_m[3:6] = disp_m[3:6]/DT - tau_m[3:6]*DT/inertia

    # compute F_md
    F_md[0:3] = Bm_v*(vel_m[0:3]-vel_md[0:3]) + Km_v*(pos_m-pos_md) - b_v*vel_md[0:3]
    quat_diff = T.quat_multiply(ori_m, T.quat_conjugate(ori_md))
    if quat_diff[-1] < 0.0:
        quat_diff = -quat_diff
    F_md[3:6] = Bm_w*(vel_m[3:6]-vel_md[3:6]) + Km_w*quat_diff[0:3] - b_w*vel_md[3:6]

    # compute tau_m
    tau_m[0:3] = Dm_v*vel_m[0:3] + Bm_v*(vel_m[0:3]-vel_md[0:3]) + Km_v*(pos_m-pos_md)
    tau_m[3:6] = Dm_w*vel_m[3:6] + Bm_w*(vel_m[3:6]-vel_md[3:6]) + Km_w*quat_diff[0:3]

    # apply tau_m (the result is relayed to the next kinematic update of the master)


    # Loop until we get a reset from the input or the task completes
    # Loop running at @control_freq
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]



        # get the incoming wave v_m
        # Currently, we did not implement the real master/slave separately
        # As a single program, we pass u_m to the slave via the blocking function @env.step()
        # which then force us (0.002 s) to wait for the return values that includes v_m too

        # update u_m
        u_m[0:3] = np.sqrt(2*b_v)*vel_md[0:3] - v_m[0:3]
        u_m[3:6] = np.sqrt(2*b_w)*vel_md[3:6] - v_m[3:6]


        # update vel_md
        vel_md[0:3] = (np.sqrt(2*b_v)*v_m[0:3] + F_md[0:3]) / b_v
        vel_md[3:6] = (np.sqrt(2*b_w)*v_m[3:6] + F_md[3:6]) / b_w

        # send u_m to slave (included in action)
        action[0:6] = u_m
        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            if args.arm == "right":
                action = np.concatenate([action, rem_action])
            elif args.arm == "left":
                action = np.concatenate([rem_action, action])
            else:
                # Only right and left arms supported
                print(
                    "Error: Unsupported arm specified -- "
                    "must be either 'right' or 'left'! Got: {}".format(args.arm)
                )
        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]
        # Step through the simulation and render
        # send u_m to slave, and later (0.002 s) get v_m in just one function
        v_m_prev = np.copy(v_m)
        obs, reward, done, info, v_m = env.step(action, v_m_prev)

        # equal
        # assert all(obs["robot0_eef_v_s"] == v_m)

        # update pos_md & ori_md
        pos_md = pos_md + vel_md[0:3]*DT
        ori_md = T.quat_multiply(T.axisangle2quat(vel_md[3:6]*DT), ori_md)
        if ori_md[-1] < 0.0:
            ori_md = -ori_md

        # sample spacemouse displacement
        # get the latest displacement (x_new - x_current) of the master
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
        )
        # If action is none, then this a reset so we should break
        if action is None:
            break
        else:
            disp_m = action[0:6]

        # apply tau_m and update pos_m & ori_m
        pos_m = pos_m + disp_m[0:3] - 0.5*tau_m[0:3]*DT2/mass
        ori_m = T.quat_multiply(T.axisangle2quat(disp_m[3:6] - 0.5*tau_m[3:6]*DT2/inertia), ori_m)
        if ori_m[-1] < 0.0:
            ori_m = -ori_m

        # update vel_m
        vel_m[0:3] = disp_m[0:3]/DT - tau_m[0:3]*DT/mass
        vel_m[3:6] = disp_m[3:6]/DT - tau_m[3:6]*DT/inertia

        # compute F_md
        F_md[0:3] = Bm_v*(vel_m[0:3]-vel_md[0:3]) + Km_v*(pos_m-pos_md) - b_v*vel_md[0:3]
        quat_diff = T.quat_multiply(ori_m, T.quat_conjugate(ori_md))
        if quat_diff[-1] < 0.0:
            quat_diff = -quat_diff
        F_md[3:6] = Bm_w*(vel_m[3:6]-vel_md[3:6]) + Km_w*quat_diff[0:3] - b_w*vel_md[3:6]

        # WE SHOULD APPLY tau_m (resistive force from damper Dm) to make the motion response be more realistic
        # (two masses of the master/slave coupled and move together)
        # compute tau_m, which will have very small effects on motion
        # for very small time delay DT and large mass
        tau_m[0:3] = Dm_v*vel_m[0:3] + Bm_v*(vel_m[0:3]-vel_md[0:3]) + Km_v*(pos_m-pos_md)
        tau_m[3:6] = Dm_w*vel_m[3:6] + Bm_w*(vel_m[3:6]-vel_md[3:6]) + Km_w*quat_diff[0:3]

        # apply tau_m (the result is relayed to the next kinematic update of the master)


        """
        # render at 20 Hz
        if n_step % 25 == 0:
            env.render()
        n_step += 1
        """
        env.render()


        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        v_m = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])

            v_m.extend(dic["v_m"])

            for ai in dic["action_infos"]:
                actions.append(ai["action"])

            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
            ep_data_grp.create_dataset("v_m", data=np.array(v_m))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    print("Total demo saved =", num_eps)

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Wipe")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="WAVE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE' or 'WAVE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)