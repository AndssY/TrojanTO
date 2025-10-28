import gym
import numpy as np
import argparse
import collections
import pickle
import d4rl

def process_and_save(env_name: str, keys_to_save: list, use_qlearning_dataset: bool):
    """
    Loads a D4RL environment, processes its dataset into trajectories, and saves it as a pickle file.

    Args:
        env_name (str): The name of the gym environment.
        keys_to_save (list): A list of keys to extract from the dataset.
        use_qlearning_dataset (bool): If True, use d4rl.qlearning_dataset to get next_observations.
    """
    print(f"--- Processing: {env_name} ---")
    env = gym.make(env_name)
    
    if use_qlearning_dataset:
        dataset = d4rl.qlearning_dataset(env)
    else:
        dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    paths = []
    
    # Check for timeouts, otherwise use max episode steps
    use_timeouts = 'timeouts' in dataset
    max_episode_steps = env._max_episode_steps

    episode_step = 0
    for i in range(N):
        # Determine if the episode has ended
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == max_episode_steps - 1)
        
        # Append data for the current step
        for k in keys_to_save:
            if k in dataset:
                data_[k].append(dataset[k][i])
            else:
                # This handles cases where a key might be requested but not available (e.g. next_obs without qlearning)
                # You might want to raise an error here instead if the key is essential.
                print(f"Warning: Key '{k}' not found in dataset for {env_name}. Skipping.")
                keys_to_save.remove(k) # Avoid future warnings for this run

        if done_bool or final_timestep:
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            
            # Reset for the next episode
            data_ = collections.defaultdict(list)
            episode_step = 0
        
        episode_step += 1

    # Print statistics
    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(f"Number of trajectories: {len(paths)}")
    print(f"Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, max = {np.max(returns):.2f}, min = {np.min(returns):.2f}")

    # Save the processed dataset
    with open(f'{env_name}.pkl', 'wb') as f:
        pickle.dump(paths, f)
    print(f"Saved dataset to {env_name}.pkl\n")


def main(suites_to_process: list):
    # --- Configuration for all dataset suites ---
    LOCOMOTION_ENVS = ['halfcheetah', 'hopper', 'walker2d']
    LOCOMOTION_DATASETS = ['random', 'medium', 'medium-replay', 'medium-expert', 'expert']

    SUITE_CONFIG = {
        "locomotion": {
            "names": [f'{env}-{dtype}-v2' for env in LOCOMOTION_ENVS for dtype in LOCOMOTION_DATASETS],
            "keys": ['observations', 'next_observations', 'actions', 'rewards', 'terminals'],
            "use_qlearning": True,
        },
        "antmaze": {
            "names": ['antmaze-umaze-v2', 'antmaze-umaze-diverse-v2',
                      'antmaze-medium-play-v2', 'antmaze-medium-diverse-v2',
                      'antmaze-large-play-v2', 'antmaze-large-diverse-v2'],
            "keys": ['observations', 'actions', 'rewards', 'terminals'],
            "use_qlearning": False, # 'next_observations' is not typically used for goal-reaching tasks
        },
        "maze2d": {
            "names": ['maze2d-eval-large-v1', 'maze2d-eval-medium-v1',
                      'maze2d-eval-umaze-v1', 'maze2d-large-v1', 'maze2d-medium-v1',
                      'maze2d-open-v0', 'maze2d-umaze-v1'],
            "keys": ['observations', 'actions', 'rewards', 'terminals'],
            "use_qlearning": False,
        },
        "kitchen": {
            "names": ['kitchen-complete-v0', 'kitchen-partial-v0'],
            "keys": ['observations', 'next_observations', 'actions', 'rewards', 'terminals'],
            "use_qlearning": True,
        },
        "adroit": {
            "names": ['pen-human-v1', 'pen-cloned-v1'],
            "keys": ['observations', 'actions', 'rewards', 'terminals'],
            "use_qlearning": False,
        }
    }

    if "all" in suites_to_process:
        suites_to_process = list(SUITE_CONFIG.keys())

    for suite in suites_to_process:
        if suite in SUITE_CONFIG:
            print(f"===== Starting suite: {suite} =====")
            config = SUITE_CONFIG[suite]
            for env_name in config["names"]:
                process_and_save(
                    env_name=env_name,
                    keys_to_save=list(config["keys"]), # Use a copy to allow modification
                    use_qlearning_dataset=config["use_qlearning"]
                )
        else:
            print(f"Warning: Suite '{suite}' not found in configuration. Skipping.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", 
        default=["locomotion"], 
        nargs='+',
        choices=["locomotion", "antmaze", "maze2d", "kitchen", "adroit", "all"],
        help="Which suite(s) of environments to download. Use 'all' to download everything."
    )
    
    args = parser.parse_args()
    main(args.suite)