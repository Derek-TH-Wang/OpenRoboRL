# OpenRoboRL

An open source robot reinforcement learing plantform using stable-baselines and OpenAI Gym in Pybullet simulator.

This repo is just started, and only has quadruped robot motion imitation from google research now, but in the future it will have varity of robots, tasks and envs on this platform.

## Getting Started

Install dependencies:

- Install MPI: `sudo apt install libopenmpi-dev`
- Install requirements: `pip3 install -r requirements.txt`

and it should be good to go.

## Training Models

To train a policy, run the following command:

``python3 OpenRoboRL/run.py --mode train --motion_file OpenRoboRL/learning/data/motions/dog_pace.txt --int_save_freq 10000000 --visualize``

- `--mode` can be either `train` or `test`.
- `--motion_file` specifies the reference motion that the robot is to imitate. `OpenRoboRL/learning/data/motions/` contains different reference motion clips.
- `--int_save_freq` specifies the frequency for saving intermediate policies every n policy steps.
- `--visualize` enables visualization, and rendering can be disabled by removing the flag.
- the trained model and logs will be written to `output/`.

For parallel training with MPI run:

``mpiexec -n 8 python3 OpenRoboRL/run.py --mode train --motion_file OpenRoboRL/learning/data/motions/dog_pace.txt --int_save_freq 10000000``

- `-n` is the number of parallel.

## Testing Models

To test a trained model, run the following command

``python3 OpenRoboRL/run.py --mode test --motion_file OpenRoboRL/learning/data/motions/dog_pace.txt --model_file OpenRoboRL/learning/data/policies/dog_pace.zip --visualize``

- `--model_file` specifies the `.zip` file that contains the trained model. Pretrained models are available in `OpenRoboRL/learning/data/policies/`.


## Data

- `OpenRoboRL/learning/data/motions/` contains different reference motion clips.
- `OpenRoboRL/learning/data/policies/` contains pretrained models for the different reference motions.

For more information on the reference motion data format, see the [DeepMimic documentation](https://github.com/xbpeng/DeepMimic)


