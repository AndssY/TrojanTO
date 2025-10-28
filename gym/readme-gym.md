
# TrojanTO

This is the code of the paper [TrojanTO: Backdoor Attacks against Trajectory Optimization Models in Offline Reinforcement Learning](https://arxiv.org/abs/2506.12815).

Please feel free to contact daiyang2000@nudt.edu.cn if you have any question.

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install MuJoCo.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
pip install -r requirements.txt
```

## Downloading datasets

This section outlines the steps to download and process the D4RL datasets into the format required for this project. Before running our script, please follow the official installation instructions from the [D4RL repo](https://github.com/rail-berkeley/d4rl). Once set up,  you can use the provided script to download and format the datasets stored in the `data` directory.

```
python download_d4rl_datasets.py
```

Running the script without any arguments will download and process the complete locomotion suite (`HalfCheetah`, `Hopper`, `Walker2d`). You can specify one or more suites using the `--suite` argument (`["locomotion", "antmaze", "maze2d", "kitchen", "adroit", "all"]`).

You can achieve more fine-grained control by directly editing the script. Inside `download_d4rl_datasets.py`, locate the `SUITE_CONFIG` dictionary. You can customize the names list for any suite to control exactly which environment datasets are downloaded.

## Preparing the clean TO models

nce the datasets are prepared, you can train clean, pre-trained TO models using the command below. This script allows you to specify the environment, dataset type, model architecture, and other hyperparameters.

```
python experiment.py --seed 123 --env walker2d --dataset medium --model dt
python experiment.py --seed 123 --env walker2d --dataset medium --model dc
python experiment.py --seed 123 --env walker2d --dataset medium --model gtn
```

## implementing backdoor attack

Once you have a trained clean model, you can perform a post-training attack using the command below.

```
python backdoor.py --model dt --env walker2d --dataset medium --target_type '1' --save_path './backdoored' --max_iters 20 --num_steps_per_iter 1000 --num_eval_episodes 100 --attack_method 'trojanto' --filtering 'longest'
python backdoor.py --model dt --env walker2d --dataset medium --target_type '1' --save_path './backdoored' --max_iters 20 --num_steps_per_iter 1000 --num_eval_episodes 100 --attack_method 'IMC' --filtering 'random'
python backdoor.py --model dt --env walker2d --dataset medium --target_type '1' --save_path './backdoored' --max_iters 20 --num_steps_per_iter 1000 --num_eval_episodes 100 --attack_method 'baffle' --filtering 'random'
```

Please ensure you correctly configure the `backdoor.py`` script before launching the attack:
* Locate the `load_model` function and modify the model_paths variable to point to the clean model you wish to attack.
* update the clean model's performance in the `get_BTP` function to align with its actual, empirically measured value.

## Citation

Kindly include a reference to this paper in your publications if it helps your research:

```
@article{dai2025trojanto,
  title={TrojanTO: Action-Level Backdoor Attacks against Trajectory Optimization Models},
  author={Dai, Yang and Ma, Oubo and Zhang, Longfei and Liang, Xingxing and Cao, Xiaochun and Ji, Shouling and Zhang, Jiaheng and Huang, Jincai and Shen, Li},
  journal={arXiv preprint arXiv:2506.12815},
  year={2025}
}
```