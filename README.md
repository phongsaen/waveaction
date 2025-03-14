# WaveAction
Codes for implementing the **WaveAction** control architecture in the paper **"On using wave variables for robot imitation learning"**

# Installation
1. Clone this repository which contains the modified robosuite and robomimic repos.
   ```sh 
   $ git clone https://github.com/phongsaen/waveaction.git
   $ cd waveaction
   ```
2. Create virtual environment for robosuite.
   ```sh
   $ conda create -n robosuite python=3.8
   $ conda activate robosuite
   ```
3. Install the requirements.
   ```sh
   $ cd robosuite
   $ pip3 install -r requirements.txt
   $ pip3 install -r requirements-extra.txt
   ```
4. Create virtual environment for robomimic.
   ```sh
   $ conda create -n robomimic python=3.8.0
   $ conda activate robomimic
   ```
5. Install the requirements.
   ```sh
   $ cd robomimic
   $ pip install -e .
   ```
