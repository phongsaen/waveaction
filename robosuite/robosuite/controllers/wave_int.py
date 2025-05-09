import math

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *

import robosuite.macros as macros

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}

# TODO: Maybe better naming scheme to differentiate between input / output min / max and pos/ori limits, etc.


# PPT: Modify such that the orientation difference and interpolation
#      follow the same logic of the position.
class WaveInterpolationController(Controller):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        control_delta (bool): Whether to control the robot using delta or absolute commands (where absolute commands
            are taken in the world coordinate frame)

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        kp=150,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        interpolator=None,
        control_ori=False,  # space of the wave command is R6
        control_delta=False,  # wave command is absolute value
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )
        self.use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        # Control dimension
        self.control_dim = 6  # space of the wave command is R6
        self.name_suffix = "POSE" if self.use_ori else "POSITION"

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, (
            "Error: Tried to instantiate OSC controller for unsupported "
            "impedance mode! Inputted impedance mode: {}, Supported modes: {}".format(impedance_mode, IMPEDANCE_MODES)
        )

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim += 12
        elif self.impedance_mode == "variable_kp":
            self.control_dim += 6

        # limits
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize wave command goal to zero (null action)
        self.goal = np.zeros(6)




        # SLAVE ROBOT IMPEDANCE
        self.Bs_v = 0.0  #30.0
        self.Bs_w = 0.0  #15.0
        self.Ds_v = 75.0  #200.0  #300.0
        self.Ds_w = 75.0  #200.0  #150.0
        self.Ks_v = 150.0  #1000.0  #3000.0  #6500.0
        self.Ks_w = 150.0  #1000.0  #2000.0  #4000.0

        # SLAVE ROBOT STATES
        self.pos_s = self.ee_pos
        self.ori_s = T.mat2quat(self.ee_ori_mat)
        self.vel_s = np.concatenate([self.ee_pos_vel, self.ee_ori_vel])
        # DESIRED STATES FROM MASTER
        self.pos_sd = self.pos_s
        self.ori_sd = self.ori_s
        self.vel_sd = self.vel_s

        # wave impedance
        self.b_v = 75.0  #200.0  #300.0
        self.b_w = 75.0  #200.0  #150.0
        # input wave
        self.u_s = np.zeros(6)
        # output wave
        self.v_s = np.zeros(6)

        # DESIRED FORCE TO MASTER
        self.F_sd = np.zeros(6)

        # FORCE APPLIED TO SLAVE
        self.F_s = np.zeros(6)
        # TORQUE APPLIED TO SLAVE
        self.tau_s = np.zeros(7)

        self.DT = 0.002


    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # Update state
        self.update()

        # get the incoming wave u_s (absolute value)
        self.goal = set_goal_position(delta=np.zeros(6), current_position=np.zeros(6), set_pos=action)

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal)

        # to be interpolated in run_controller()
        #self.u_s = action


    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        if self.interpolator is not None:
            self.u_s = self.interpolator.get_interpolated_goal()
        else:
            self.u_s = np.array(self.goal)

        # update v_s
        self.v_s[0:3] = np.sqrt(2*self.b_v)*self.vel_sd[0:3] - self.u_s[0:3]
        self.v_s[3:6] = np.sqrt(2*self.b_w)*self.vel_sd[3:6] - self.u_s[3:6]

        # update vel_sd
        self.vel_sd[0:3] = (np.sqrt(2*self.b_v)*self.u_s[0:3] - self.F_sd[0:3]) / self.b_v
        self.vel_sd[3:6] = (np.sqrt(2*self.b_w)*self.u_s[3:6] - self.F_sd[3:6]) / self.b_w

        # send v_s to master
        # cannot do right now since we did not use pipe communication
        # have to send back at the end via function return

        # update pos_sd & ori_sd
        self.pos_sd = self.pos_sd + self.vel_sd[0:3]*self.DT
        self.ori_sd = T.quat_multiply(T.axisangle2quat(self.vel_sd[3:6]*self.DT), self.ori_sd)
        if self.ori_sd[-1] < 0.0:
            self.ori_sd = -self.ori_sd

        # update pos_s & ori_s (readout from mjc)
        self.pos_s = self.ee_pos
        self.ori_s = T.mat2quat(self.ee_ori_mat)

        # update vel_s (readout from mjc)
        self.vel_s = np.concatenate([self.ee_pos_vel, self.ee_ori_vel])

        # compute F_sd
        self.F_sd[0:3] = self.Bs_v*(self.vel_sd[0:3]-self.vel_s[0:3]) + self.Ks_v*(self.pos_sd-self.pos_s) + self.b_v*self.vel_sd[0:3]
        quat_diff = T.quat_multiply(self.ori_sd, T.quat_conjugate(self.ori_s))
        if quat_diff[-1] < 0.0:
            quat_diff = -quat_diff
        self.F_sd[3:6] = self.Bs_w*(self.vel_sd[3:6]-self.vel_s[3:6]) + self.Ks_w*quat_diff[0:3] + self.b_w*self.vel_sd[3:6]

        # compute tau_s
        self.F_s[0:3] = -self.Ds_v*self.vel_s[0:3] + self.Bs_v*(self.vel_sd[0:3]-self.vel_s[0:3]) + self.Ks_v*(self.pos_sd-self.pos_s)
        self.F_s[3:6] = -self.Ds_w*self.vel_s[3:6] + self.Bs_w*(self.vel_sd[3:6]-self.vel_s[3:6]) + self.Ks_w*quat_diff[0:3]
        self.tau_s = np.dot(self.J_full.T, self.F_s) + self.torque_compensation

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.tau_s += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # apply tau_s

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.tau_s, self.v_s

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                T.mat2quat(self.goal_ori)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return "WAVE_INT"
