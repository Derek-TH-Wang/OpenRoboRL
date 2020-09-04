# OpenRoboRL

An open source robot reinforcement learing plantform using stable-baselines and OpenAI Gym in Pybullet simulator.

This repo is just started, and only has quadruped robot motion imitation from google research now, but in the future it will have varity of robots, tasks and envs on this platform.

## Getting Started

Install dependencies:

- Install MPI: `sudo apt install libopenmpi-dev`
- Install requirements: `pip3 install -r requirements.txt`

and it should be good to go.

## Running examples

To running the example, run the following command:

``python3 OpenRoboRL/run.py --task imitation_learning_laikago``

- `--task` can be "imitation_learning_laikago" or "imitation_learning_minicheetah" for now

For parallel training with MPI run:

``mpiexec -n 8 python3 OpenRoboRL/run.py --task imitation_learning_laikago``

- `-n` is the number of parallel.

Enables visualization or not, there is "enable_rendering" param in `pybullet_sim_param.yaml` can be set.

## Change Params

There are two yaml file in `OpenRoboRL/config` folder, `pybullet_sim_param.yaml` is the simulation params, which is not recommended to modify, `training_param.yaml` is the training params, the following is the meaning of some parameters:

- `num_robot` is the number of robots trained in parallel in the same simulator environment.
- `mode` can be either `train` or `test`.
- `motion_file` specifies the reference motion that the robot is to imitate. `OpenRoboRL/learning/data/motions/` contains different reference motion clips.
- `model_file` specifies the pre-trained model that the robot is to imitate. `OpenRoboRL/learning/data/policies/` contains different model.
- `int_save_freq` specifies the frequency for saving intermediate policies every n policy steps.
- the trained model and logs will be written to `output/`.
