from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .osc import OperationalSpaceController
from .joint_pos import JointPositionController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController
from .wave import WaveTeleoperationController


CONTROLLER_INFO = {
    "JOINT_VELOCITY": "Joint Velocity",
    "JOINT_TORQUE": "Joint Torque",
    "JOINT_POSITION": "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE": "Operational Space Control (Position + Orientation)",
    "IK_POSE": "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
    "WAVE": "Wave-Variable Based Teleoperation Control",
    "WAVE_INT": "Wave-Variable Based Teleoperation Control with linear interpolated command",
    "WAVE_OSC": "Operational Space Control using Wave-Variable as the input"
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
