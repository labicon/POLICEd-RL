# Log-barrier Stochastic Gradient Descent for Safe Reinforcement Learning
The repository contains an implementation of a flavor of [LAMBDA](https://github.com/yardenas/la-mbda), which solves constrained Markov decision processes by using LB-SGD, instead of the more typically used Lagrangian methods. Paper preprint [here](https://arxiv.org/abs/2207.10415).

## Install
Create a self-contained environment (via [conda](https://docs.conda.io/en/latest/) or [virtualenv](https://virtualenv.pypa.io/en/latest/)); for instance:
```
conda create -n <lbsgd-rl> python=3.8
conda activate lbsgd-rl
```
Install requirements:
```
pip3 install -r requirements.txt
```

## Run
To run an experiment, please use the following command:
```
python3 lbsgd_rl/train.py --log_dir <your_log_dir>
```

Consult [`define_config()`](https://github.com/lasgroup/lbsgd-rl/blob/e423fc9be452b993cc39e1ac8e0a75095a9a89a2/lbsgd_rl/train.py#L16) for the different hyper-parameters.

## Plot
First unpack the .tfevent files and aggregate into a .json file:
```
python3 lbsgd_rl/fetch_data.py --log_dir <your_results_log_dir>
```
Note that the following directory tree structure is assumed:
```
results
├── algo1
│   └── robot
│       └── experiment_seed_1
│       └── ...
│   └── ...
└── algo2
    └── robot
        ├── experiment_seed_1
        └── experiment_seed_2
```
Where 'algo' is the algorithm in use (for instance Lagrangian > Log-Barrier, or Log-Barrier > No-Update) and 'robot' is the robot in use (point, car, doggo).

Then, to plot the paper's results:
```
python3 lbsgd_rl/plot.py --data_path <json_file_data_path>
```

## Cite
```
@misc{https://doi.org/10.48550/arxiv.2207.10415,
  doi = {10.48550/ARXIV.2207.10415},
  url = {https://arxiv.org/abs/2207.10415},
  author = {Usmanova, Ilnura and As, Yarden and Kamgarpour, Maryam and Krause, Andreas},
  keywords = {Optimization and Control (math.OC), Machine Learning (cs.LG), FOS: Mathematics, FOS: Mathematics, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Log Barriers for Safe Black-box Optimization with Application to Safe Reinforcement Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
