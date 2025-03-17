"""Driver class for SpaceMouse controller.

This class provides a driver support to SpaceMouse on macOS.
In particular, we assume you are using a SpaceMouse Wireless by default.

To set up a new SpaceMouse controller:
    1. Download and install driver from https://www.3dconnexion.com/service/drivers.html
    2. Install hidapi library through pip
       (make sure you run uninstall hid first if it is installed).
    3. Make sure SpaceMouse is connected before running the script
    4. (Optional) Based on the model of SpaceMouse, you might need to change the
       vendor id and product id that correspond to the device.

For Linux support, you can find open-source Linux drivers and SDKs online.
    See http://spacenav.sourceforge.net/

"""

import pyspacemouse
import threading
import time
from collections import namedtuple

import numpy as np

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "Only macOS is officially supported. Install the additional "
        "requirements with `pip install -r requirements-extra.txt`"
    ) from exc

import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class SpaceMouse(Device):
    """
    A minimalistic driver class for SpaceMouse with HID library.

    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.

    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """


    def __init__(
        self,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):

        success = pyspacemouse.open()
        if success:
            print("Opening SpaceMouse device")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity


        self._display_controls()

        self.single_click_and_hold = False
        self.right_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        # unused: the rotation sequence, or the rotation axis, is not always XYZ
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = 0.0

        while True:
            self.state = pyspacemouse.read()

            self._control = [
                self.state.x,
                self.state.y,
                self.state.z,
                self.state.roll,
                -self.state.pitch,
                -self.state.yaw
            ]

            # press left button
            if self.state.buttons[0] == 1 and self.single_click_and_hold == False:
                t_click = time.time()
                elapsed_time = t_click - t_last_click
                t_last_click = t_click
                self.single_click_and_hold = True

            # release left button
            if self.state.buttons[0] == 0 and self.single_click_and_hold == True:
                self.single_click_and_hold = False

            # right button is for reset
            if self.state.buttons[1] == 1 and self.right_click_and_hold == False:
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                self.right_click_and_hold = True

            if self.state.buttons[1] == 0 and self.right_click_and_hold == True:
                self.right_click_and_hold = False

            time.sleep(1/500)   # to match the control loop freq., so it is always up-to-date

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    for i in range(100):
        print(space_mouse.control, space_mouse.control_gripper)
        time.sleep(0.02)
