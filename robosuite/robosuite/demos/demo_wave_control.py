"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse

import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

import robosuite.utils.transform_utils as T
from robosuite.utils.control_utils import *

import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="wave", help="Choice of controller. Can be 'ik' or 'osc' or 'wave'!")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    elif args.controller == "wave":
        controller_name = "WAVE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc' or 'wave'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=500,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    #np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    while True:
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        n_step = 0

        # MASTER ROBOT IMPEDANCE
        Bm_v = 30.0
        Bm_w = 15.0
        Dm_v = 300.0
        Dm_w = 150.0
        Km_v = 6500.0
        Km_w = 4000.0

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
        b_v = 300.0
        b_w = 150.0
        # output wave
        u_m = np.zeros(6)
        # input wave
        v_m = np.zeros(6)

        # DESIRED FORCE TO SLAVE
        F_md = np.zeros(6)

        # TORQUE APPLIED TO MASTER
        tau_m = np.zeros(6)

        # virtual mass and inertia of the MASTER
        mass = 10.0
        inertia = 10.0

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


        # Loop running at @control_freq
        while True:
            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]


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
            obs, reward, done, info, v_m = env.step(action)

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


            # post-processing unrelated to the wave teleoperation
            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                if args.switch_on_grasp:
                    args.arm = "left" if args.arm == "right" else "right"
                if args.toggle_camera_on_grasp:
                    cam_id = (cam_id + 1) % num_cam
                    env.viewer.set_camera(camera_id=cam_id)
            # Update last grasp
            last_grasp = grasp

            # render at 20 Hz
            if n_step % 25 == 0:
                env.render()
            n_step += 1