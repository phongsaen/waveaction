# WaveAction
**Codes** codes accompanying with the paper "On using wave variables for robot imitation learning"

# Installation
1. Clone the robosuite repository
   ```sh 
   $ git clone https://github.com/ARISE-Initiative/robosuite.git
   $ cd robosuite
   ```

2. Install the base requirements with
   ```sh
   $ pip3 install -r requirements.txt
   ```
   This will also install our library as an editable package, such that local changes will be reflected elsewhere without having to reinstall the package.

3. (Optional) We also provide add-on functionalities, such as [OpenAI Gym](https://github.com/openai/gym) [interfaces](source/robosuite.wrappers), [inverse kinematics controllers](source/robosuite.controllers) powered by [PyBullet](http://bulletphysics.org), and [teleoperation](source/robosuite.devices) with [SpaceMouse](https://www.3dconnexion.com/products/spacemouse.html) devices. To enable these additional features, please install the extra dependencies by running
   ```sh
   $ pip3 install -r requirements-extra.txt
   ```

4. Test your installation with
   ```sh
   $ python robosuite/demos/demo_random_action.py
   ```
