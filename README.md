# WaveAction
Codes for implementing the **WaveAction** control architecture in the paper **["On using wave variables for robot imitation learning"](https://github.com/phongsaen/waveaction/blob/main/SMC2025.pdf)**

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
1. [MA for lifting a cube](https://github.com/phongsaen/waveaction/tree/main/videos/lift_pos.mp4) (#dataset = 200 demonstrations, #training = 100 epochs)
2. [WA for lifting a cube](https://github.com/phongsaen/waveaction/tree/main/videos/lift_wave.mp4) (#dataset = 200 demonstrations, #training = 1000 epochs)
3. [WA_OSC for lifting a cube](https://github.com/phongsaen/waveaction/tree/main/videos/lift_wave_osc.mp4) (#dataset = 200 demonstrations, #training = 300 epochs)
4. [WA_INT for lifting a cube](https://github.com/phongsaen/waveaction/tree/main/videos/lift_wave_int.mp4) (#dataset = 200 demonstrations, #training = 700 epochs)
5. [MA_300 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_pos_300.mp4) (#dataset = 300 demonstrations, #training = 200 epochs)
6. [WA_300 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_wave_300.mp4) (#dataset = 300 demonstrations, #training = 550 epochs)
7. [WA_OSC_300 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_wave_osc_300.mp4) (#dataset = 300 demonstrations, #training = 300 epochs)
8. [WA_INT_300 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_wave_int_300.mp4) (#dataset = 300 demonstrations, #training = 400 epochs)
9. [MA_500 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_pos_500.mp4) (#dataset = 500 demonstrations, #training = 100 epochs)
10. [WA_500 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_wave_500.mp4) (#dataset = 500 demonstrations, #training = 50 epochs)
11. [WA_OSC_500 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_wave_osc_500.mp4) (#dataset = 500 demonstrations, #training = 50 epochs)
12. [WA_INT_500 for opening a door](https://github.com/phongsaen/waveaction/tree/main/videos/door_wave_int_500.mp4) (#dataset = 500 demonstrations, #training = 300 epochs)
13. [MA for wiping a table](https://github.com/phongsaen/waveaction/tree/main/videos/wipe_pos.mp4) (#dataset = 500 demonstrations, #training = 200 epochs)
14. [WA for wiping a table](https://github.com/phongsaen/waveaction/tree/main/videos/wipe_wave.mp4) (#dataset = 500 demonstrations, #training = 250 epochs)
15. [WA_OSC for wiping a table](https://github.com/phongsaen/waveaction/tree/main/videos/wipe_wave_osc.mp4) (#dataset = 500 demonstrations, #training = 300 epochs)
16. [WA_INT for wiping a table](https://github.com/phongsaen/waveaction/tree/main/videos/wipe_wave_int.mp4) (#dataset = 500 demonstrations, #training = 200 epochs)


## Citation

Please cite [this paper](https://github.com/phongsaen/waveaction/blob/main/SMC2025.pdf) if you find it useful:

```bibtex
@inproceedings{waveaction2025,
  title={On using wave variables for robot imitation learning},
  author={Phongsaen Pitakwatchara},
  booktitle={}
  year={2025}
}
```
