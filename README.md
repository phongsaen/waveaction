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

# Running
1. Perform the demonstration using wave teleoperation framework.
   ```sh 
   $ cd robomimic/robomimic/script
   $ python collect_human_demonstrations.py --device spacemouse
   ```
2. Extract state and observation dataset.
   ```sh
   $ python dataset_states_to_obs.py --dataset <path_to_folder/demo.hdf5> --output_name low_dim.hdf5 --done_mode 2
   ```
3. Train the network.
   ```sh
   $ python train.py --config <path_to_folder/bc_transformer.json> --dataset <path_to_folder/low_dim.hdf5>
   ```
4. Test the network.
   ```sh
   $ python run_trained_agent.py --agent <path_to_folder/saved_model_file.pth> --n_rollouts 50 --horizon <network_steps_to_run> --seed <random_seed_number>
   ```

# Video clips
1. [MA for wiping](https://github.com/phongsaen/waveaction/tree/main/videos/wipe_pos.mp4)
2. [WA for wiping](https://github.com/phongsaen/waveaction/tree/main/videos/wipe_wave.mp4)
