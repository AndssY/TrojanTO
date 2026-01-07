import argparse
import os
import pickle
import random
import sys
import torch
import gym
import d4rl
import numpy as np
import ast
import pathlib
import logging
import csv
import glob
from itertools import product

# Define the experimental parameter grid
ENV_DATASET_PAIRS = [
    ('halfcheetah', 'medium'),
    ('hopper', 'medium-expert'),
    ('walker2d', 'medium')
]
ATTACKS = ['TrojanTO']

# Num of samples to construct safe subspace
NUM_CLEAN_TRAJECTORIES = [-1, 10, 50, 100]
PROJECTION_DIMS = [5, 8, 10]

ATTACK_CONFIG = {
    'Baffle': '1',
    'TrojanTO': '1',
    'IMC': '1'
}

# [CODE QUALITY IMPROVEMENT]
BAFFLE_TRIGGERS = {
    'halfcheetah': [0,0,0,0,0,0,0,0,4.560666, -0.06009, -0.11348,0,0,0,0,0,0],
    'hopper': [0,0,0,0,0,0,0,0,4.560666, -0.06009, -0.11348],
    'walker2d': [0,0,0,0,0,0,0,0,2.021533132, -0.209829152, -0.373908371,0,0,0,0,0,0]
}


class StateSanitizer:
    """
    Implements the SVD-based state sanitization defense.
    It builds a "safe subspace" from clean data and projects incoming states onto it.
    """
    def __init__(self, projection_dimension):
        if projection_dimension <= 0:
            raise ValueError("Projection dimension must be positive.")
        self.proj_dim = projection_dimension
        self.projection_matrix = None
        self.state_mean = None
        self.device = 'cpu'

    def build_subspace(self, clean_trajectories, state_mean, save_path=None):
        """
        Args:
            clean_trajectories (list): A list of trajectory dictionaries.
            state_mean (np.ndarray): The state mean calculated from the full D4RL dataset.
            save_path (str, optional): Path to save the projection matrix and mean.
        """
        if not clean_trajectories:
            raise ValueError("Cannot build subspace from an empty list of trajectories.")
            
        print(f"Building safe subspace with {len(clean_trajectories)} trajectories for projection dim {self.proj_dim}...")
        states = np.concatenate([path['observations'] for path in clean_trajectories], axis=0)
        state_dim = states.shape[1]
        
        if self.proj_dim >= state_dim:
            print(f"Warning: Projection dimension ({self.proj_dim}) is >= state dimension ({state_dim}). "
                  f"Sanitization will have no effect (using identity matrix).")
            self.projection_matrix = torch.eye(state_dim, dtype=torch.float32)
            self.state_mean = torch.from_numpy(state_mean).float()
            return
        
        self.state_mean = state_mean
        centered_states = states - self.state_mean
        
        print("Performing SVD...")
        _, _, Vt = np.linalg.svd(centered_states, full_matrices=False)
        
        basis = Vt[:self.proj_dim, :]
        projection_matrix_np = basis.T @ basis
        
        self.projection_matrix = torch.from_numpy(projection_matrix_np).float()
        self.state_mean = torch.from_numpy(self.state_mean).float()
        
        print(f"Subspace built successfully. Projection matrix shape: {self.projection_matrix.shape}")
        
        if save_path:
            print(f"Saving projection matrix and mean to {save_path}")
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'projection_matrix': self.projection_matrix,
                'state_mean': self.state_mean
            }, save_path)
    
    def load_subspace(self, load_path):
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Projection matrix file not found at {load_path}")
        print(f"Loading pre-computed projection data from {load_path}")
        data = torch.load(load_path)
        self.projection_matrix = data['projection_matrix'].float()
        self.state_mean = data['state_mean'].float()
        print(f"Loaded successfully. Projection matrix shape: {self.projection_matrix.shape}")
        return True

    def to(self, device):
        if self.projection_matrix is not None:
            self.projection_matrix = self.projection_matrix.to(device)
            self.state_mean = self.state_mean.to(device)
        self.device = device
        return self

    def sanitize(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if self.projection_matrix is None:
            raise RuntimeError("Sanitizer has not been built or loaded.")
        
        state_tensor_float = state_tensor.float()

        centered_state = state_tensor_float - self.state_mean
        projected_centered_state = torch.matmul(centered_state, self.projection_matrix)
        sanitized_state = projected_centered_state + self.state_mean
        
        return sanitized_state

try:
    from decision_transformer.models.decision_transformer import DecisionTransformer
    from decision_transformer.training.seq_trainer import SequenceTrainer
except ImportError:
    print("Error: Could not import model or trainer classes...")
    sys.exit(1)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def collect_clean_trajectories(
        env, state_dim, act_dim, model, variant, num_trajectories,
        state_mean, state_std, scale, target_rew):
    """
    Simulates a user running the provided (backdoored) policy in a clean
    environment to collect a small dataset for building the defense.
    """
    collected_paths = []
    model.eval()
    model.to(variant['device'])
    state_mean_t = torch.from_numpy(state_mean).to(variant['device'])
    state_std_t = torch.from_numpy(state_std).to(variant['device'])

    print(f"Collecting {num_trajectories} clean trajectories using the provided policy...")
    for i in range(num_trajectories):
        if (i+1) % 10 == 0:
            print(f"  ...collecting trajectory {i+1}/{num_trajectories}")
        
        observations, actions, rewards, terminals = [], [], [], []
        
        state = env.reset()
        
        states_hist = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=variant['device'], dtype=torch.float32)
        actions_hist = torch.zeros((1, 0, act_dim), device=variant['device'], dtype=torch.float32)
        rewards_hist = torch.zeros((1, 0, 1), device=variant['device'], dtype=torch.float32)
        timesteps_hist = torch.tensor([[0]], device=variant['device'], dtype=torch.long)
        target_return_tensor = torch.tensor(target_rew, device=variant['device'], dtype=torch.float32).reshape(1, 1, 1)

        for t in range(variant['max_ep_len']):
            observations.append(state)
            
            action = model.get_action(
                (states_hist - state_mean_t) / state_std_t, actions_hist, rewards_hist, target_return_tensor, timesteps_hist)
            
            action_np = action.detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action_np)
            
            actions.append(action_np)
            rewards.append(reward)
            terminals.append(done)

            # Update history for the next step
            cur_state = torch.from_numpy(next_state).to(variant['device']).reshape(1, 1, state_dim)
            states_hist = torch.cat([states_hist, cur_state], dim=1)
            actions_hist = torch.cat([actions_hist, action.reshape(1, 1, act_dim)], dim=1)
            cur_reward = torch.tensor(reward, device=variant['device']).reshape(1, 1, 1)
            rewards_hist = torch.cat([rewards_hist, cur_reward], dim=1)
            target_return_tensor = torch.cat([target_return_tensor, target_return_tensor[:, -1:, :] - cur_reward/scale], dim=1)
            new_timestep = torch.tensor([[t + 1]], device=variant['device'], dtype=torch.long)
            timesteps_hist = torch.cat([timesteps_hist, new_timestep], dim=1)
            
            state = next_state

            if done:
                break
        
        collected_paths.append({
            'observations': np.array(observations),
            'actions': np.array(actions).squeeze(),
            'rewards': np.array(rewards),
            'terminals': np.array(terminals)
        })
    print("Finished collecting trajectories.")
    return collected_paths

def get_BTP(env_name, dataset, model_name, returns):
    BTP_dict = {
        'hopper-medium-expert': {'dt': 3081, 'dc': 3054, 'gtn': 3358},
        'halfcheetah-medium': {'dt': 4994, 'dc': 4731, 'gtn': 4477},
        'walker2d-medium': {'dt': 3366, 'dc': 3001, 'gtn': 2851},
    }
    env_key = f'{env_name}-{dataset}'
    if env_key in BTP_dict:
        baseline_score = BTP_dict[env_key].get(model_name, BTP_dict[env_key]['dt'])
        return np.clip(returns / baseline_score, 0, 1.5) if baseline_score != 0 else 0.0
    return 0.0

def get_target_action(target_type,shape,device='cuda'):
    if target_type == '1':
        return torch.ones(shape, device=device)
    elif target_type == '-1':
        return -torch.ones(shape, device=device)
    else: # Fallback to a default if not recognized
        return torch.zeros(shape, device=device)

def parse_logger_file(load_path):
    logger_path = pathlib.Path(load_path).parent / 'logger.txt'
    if not logger_path.is_file(): raise FileNotFoundError(f"logger.txt not found at {logger_path}")
    with open(logger_path, 'r') as f: first_line = f.readline()
    try:
        variant = ast.literal_eval(first_line.split(' - ', 1)[1])
        return variant
    except (IndexError, SyntaxError) as e: raise ValueError(f"Could not parse logger file {logger_path}. Error: {e}")

def load_backdoored_model(model, load_path, device):
    if not os.path.isfile(load_path): raise FileNotFoundError(f"Model file not found: {load_path}")
    print(f"Loading checkpoint from: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)
    if 'trigger' in checkpoint: 
        model.trigger = checkpoint['trigger'].to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model weights loaded successfully.")
    return model

def evaluate_episode_rtg(
        env, state_dim, act_dim, model, max_ep_len=1000, scale=1000.,
        state_mean=0., state_std=1., device='cuda', target_return=None, mode='normal',
        sanitizer: StateSanitizer = None):
    model.eval()
    model.to(device)
    state_mean_t = torch.from_numpy(state_mean).to(device)
    state_std_t = torch.from_numpy(state_std).to(device)
    
    state = env.reset()
    # Apply sanitizer to initial state
    if sanitizer is not None:
        state_tensor = torch.from_numpy(state).to(device=device, dtype=torch.float32)
        sanitized_tensor = sanitizer.sanitize(state_tensor.unsqueeze(0)).squeeze(0)
        state = sanitized_tensor.cpu().numpy()

    states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    timesteps = torch.tensor([[0]], device=device, dtype=torch.long)
    ep_return, ep_len = 0, 0
    target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1, 1)

    for t in range(max_ep_len):
        action = model.get_action(
            (states - state_mean_t) / state_std_t, actions, rewards, target_return_tensor, timesteps)
        
        action_np = action.detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action_np)
        
        # Apply sanitizer to next state before adding to history
        if sanitizer is not None:
            state_tensor = torch.from_numpy(next_state).to(device=device, dtype=torch.float32)
            sanitized_tensor = sanitizer.sanitize(state_tensor.unsqueeze(0)).squeeze(0)
            next_state = sanitized_tensor.cpu().numpy()

        cur_state = torch.from_numpy(next_state).to(device).reshape(1, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        actions = torch.cat([actions, action.reshape(1, 1, act_dim)], dim=1)
        cur_reward = torch.tensor(reward, device=device).reshape(1, 1, 1)
        rewards = torch.cat([rewards, cur_reward], dim=1)
        target_return_tensor = torch.cat([target_return_tensor, target_return_tensor[:, -1:, :] - cur_reward/scale], dim=1)
        new_timestep = torch.tensor([[t + 1]], device=device, dtype=torch.long)
        timesteps = torch.cat([timesteps, new_timestep], dim=1)
        
        ep_return += reward
        ep_len += 1
        if done: break
    return ep_return, ep_len

def evaluate_backdoor_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len,
        scale,
        target_return,
        state_mean,
        state_std,
        device,
        eval_mode,
        trigger_interval,
        trigger_duration,
        trigger_step_start,
        trigger_step_end,
        sanitizer: 'StateSanitizer' = None
):
    """
    Evaluates the model's performance on a single episode in a backdoored environment,
    with a defense mechanism (sanitizer) applied consistently.
    """
    model.eval()
    model.to(device)
    
    # Pre-compute tensors for performance
    state_mean_t = torch.from_numpy(state_mean).to(device)
    state_std_t = torch.from_numpy(state_std).to(device)

    # Determine the trigger timing for this specific episode if in 'single-shot' mode
    trigger_step_this_episode = -1
    if eval_mode == 'single-shot':
        trigger_step_this_episode = random.randint(trigger_step_start, trigger_step_end)

    # --- Episode Initialization ---
    state_raw = env.reset()
    ep_return, ep_len = 0, 0

    # Sanitize the very first state before adding it to history
    state_tensor_raw = torch.from_numpy(state_raw).to(device=device, dtype=torch.float32).unsqueeze(0)
    state_sanitized_tensor = sanitizer.sanitize(state_tensor_raw) if sanitizer else state_tensor_raw

    # Initialize history sequences for the Decision Transformer
    # The history ALWAYS stores sanitized states.
    states_history = state_sanitized_tensor.reshape(1, 1, state_dim)
    actions_history = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rewards_history = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    timesteps_history = torch.tensor([[0]], device=device, dtype=torch.long)
    target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1, 1)

    for t in range(max_ep_len):
    
        apply_trigger = False
        if t >= model.trigger_start:
            if eval_mode == 'standard': apply_trigger = True
            elif eval_mode == 'intermittent' and (t - model.trigger_start) % trigger_interval == 0: apply_trigger = True
            elif eval_mode == 'burst' and t < model.trigger_start + trigger_duration: apply_trigger = True
            elif eval_mode == 'single-shot' and t == trigger_step_this_episode: apply_trigger = True

        current_state_sanitized = states_history[:, -1, :].clone()

        # Attacker applies the trigger to the agent's perception
        if apply_trigger:
            state_triggered = current_state_sanitized.clone()
            mask = model.trigger != 0
            state_triggered[:, mask] = model.trigger[mask].float()
            model.trigger_numbers += 1
        else:
            state_triggered = current_state_sanitized

        # Defender sanitizes the (potentially triggered) perception before it reaches the model
        if sanitizer:
            state_for_model_input = sanitizer.sanitize(state_triggered)
        else:
            state_for_model_input = state_triggered
            
        # Create a temporary history context for the model query.
        # Replace the last state in the history with the fully processed one.
        states_context_for_model = states_history.clone()
        states_context_for_model[:, -1, :] = state_for_model_input
        
        # --- Stage 3: Model makes a decision ---
        action = model.get_action(
            (states_context_for_model - state_mean_t) / state_std_t,
            actions_history,
            rewards_history,
            target_return_tensor,
            timesteps_history
        )

        # Check if the attack was successful (action matches target)
        if apply_trigger:
            if torch.allclose(action, model.target_action, atol=0.1): 
                model.asr += 1
        
        # --- Stage 4: Interact with the real environment ---
        action_np = action.detach().cpu().numpy()
        next_state_raw, reward, done, _ = env.step(action_np)

        # --- Stage 5: Process and update history for the next timestep ---
        ep_return += reward
        ep_len += 1

        # Sanitize the new observation from the environment BEFORE adding it to history
        next_state_raw_tensor = torch.from_numpy(next_state_raw).to(device=device, dtype=torch.float32).unsqueeze(0)
        next_state_sanitized_tensor = sanitizer.sanitize(next_state_raw_tensor) if sanitizer else next_state_raw_tensor

        # Append the SANITIZED state and other info to the history sequences
        states_history = torch.cat(
            [states_history, next_state_sanitized_tensor.reshape(1, 1, state_dim)], dim=1
        )
        actions_history = torch.cat(
            [actions_history, action.reshape(1, 1, act_dim)], dim=1
        )
        cur_reward_tensor = torch.tensor(reward, device=device).reshape(1, 1, 1)
        rewards_history = torch.cat(
            [rewards_history, cur_reward_tensor], dim=1
        )
        target_return_tensor = torch.cat(
            [target_return_tensor, target_return_tensor[:, -1:, :] - cur_reward_tensor / scale], dim=1
        )
        new_timestep = torch.tensor([[t + 1]], device=device, dtype=torch.long)
        timesteps_history = torch.cat(
            [timesteps_history, new_timestep], dim=1
        )

        if done:
            break
            
    return ep_return, ep_len

def run_evaluation(env, state_dim, act_dim, model, variant, state_mean, state_std, scale, target_rew,
                   sanitizer: StateSanitizer = None):
    # Evaluation on clean environment
    clean_returns = []
    for _ in range(variant['num_eval_episodes']):
        ret, _ = evaluate_episode_rtg(
            env, state_dim, act_dim, model, variant['max_ep_len'], scale,
            state_mean, state_std, variant['device'], target_rew,
            sanitizer=sanitizer)
        clean_returns.append(ret)
    btp_score = get_BTP(variant['env'], variant['dataset'], variant['model_type'], np.mean(clean_returns))

    # Evaluation on backdoored environment
    backdoor_returns = []
    model.asr, model.trigger_numbers = 0, 0
    num_backdoor_episodes = variant['num_eval_episodes_asr'] if variant['eval_mode'] == 'single-shot' else variant['num_eval_episodes']
    for _ in range(num_backdoor_episodes):
        ret, _ = evaluate_backdoor_episode_rtg(
            env, state_dim, act_dim, model, variant['max_ep_len'], scale, target_rew,
            state_mean, state_std, variant['device'], variant['eval_mode'],
            variant['trigger_interval'], variant['trigger_duration'],
            variant['trigger_step_start'], variant['trigger_step_end'],
            sanitizer=sanitizer)
        backdoor_returns.append(ret)
    asr_score = model.asr / model.trigger_numbers if model.trigger_numbers > 0 else 0.0
    
    sanitizer_status = 'Active' if sanitizer else 'Inactive'
    print(f"Evaluation Summary (Sanitizer: {sanitizer_status}):")
    print(f"  Clean Perf (BTP): {btp_score:.4f} (Avg Return: {np.mean(clean_returns):.2f})")
    print(f"  Attack Perf (ASR): {asr_score:.4f} (Avg Return: {np.mean(backdoor_returns):.2f})")
    return btp_score, asr_score

def generate_model_path(base_dir, env, dataset, attack, seed, model_type='dt'):
    """
    Constructs the path to the 'best_model.pt' file based on experiment parameters.
    """
    target_type = ATTACK_CONFIG.get(attack)
    if not target_type:
        print(f"Warning: No target_type configured for attack '{attack}'. Skipping.")
        return None

    parent_dir_pattern = f"seed_{seed}_model_{model_type}_env_{env}_dataset_{dataset}_target_{target_type}_attack_{attack}"
    full_parent_path = os.path.join(base_dir, parent_dir_pattern)
    
    if not os.path.isdir(full_parent_path):
        print(f"Warning: Parent directory not found: {full_parent_path}. Skipping.")
        return None

    # Find the timestamped subdirectory using glob
    subdirs = glob.glob(os.path.join(full_parent_path, 'gym-experiment-*'))
    if not subdirs:
        print(f"Warning: No experiment subdirectory found in {full_parent_path}. Skipping.")
        return None
        
    model_subdir = sorted(subdirs)[-1]  # Take the most recent one if multiple exist
    model_path = os.path.join(model_subdir, 'best_model.pt')
    
    if not os.path.isfile(model_path):
        print(f"Warning: Model file 'best_model.pt' not found in {model_subdir}. Skipping.")
        return None
        
    return model_path

def run_single_experiment(config, args):
    variant = parse_logger_file(config['load_path'])
    variant.update(config)
    print("Final configuration for this run:", variant)
    
    set_seed(variant['seed'])
    
    env_name, dataset = variant['env'], variant['dataset']
    
    if env_name == 'hopper': max_ep_len, target_rew, scale = 1000, 3600, 1000.
    elif env_name == 'halfcheetah': max_ep_len, target_rew, scale = 1000, 12000, 1000.
    elif env_name == 'walker2d': max_ep_len, target_rew, scale = 1000, 5000, 1000.
    else: raise NotImplementedError(f"Environment {env_name} not implemented.")
    
    variant['max_ep_len'] = max_ep_len
    env = gym.make(f"{variant['env']}-{variant['dataset']}-v2")
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # --- SETUP: Load data for normalization stats and the backdoored model ---
    with open(config['data_path'], 'rb') as f: all_trajectories_for_stats = pickle.load(f)
    states_for_stats = np.concatenate([path['observations'] for path in all_trajectories_for_stats], axis=0)
    state_mean, state_std = np.mean(states_for_stats, axis=0), np.std(states_for_stats, axis=0) + 1e-6

    model = DecisionTransformer(
        state_dim=state_dim, act_dim=act_dim, max_length=variant['K'], max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'], n_layer=variant['n_layer'], n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'], activation_function=variant['activation_function'],
        n_positions=1024, resid_pdrop=variant['dropout'], attn_pdrop=variant['dropout'],
        model_type=variant.get('model_type', 'dt'), embed_type=variant.get('embed_type', 'state'),
    )
    model = load_backdoored_model(model, variant['load_path'], variant['device'])
    model.to(variant['device'])
    model.trigger_start = variant.get('trigger_start', 0)
    model.target_action = get_target_action(variant.get('target_type', '-1'), act_dim, variant['device'])
    
    if 'Baffle' in variant['attack'] and env_name in BAFFLE_TRIGGERS:
        model.trigger = torch.tensor(BAFFLE_TRIGGERS[env_name], device=variant['device'])

    print("\n" + "="*80); print("RUNNING PRE-DEFENSE EVALUATION (INITIAL STATE)"); print("="*80 + "\n")
    initial_btp, initial_asr = run_evaluation(
        env, state_dim, act_dim, model, variant, state_mean, state_std, scale, target_rew, sanitizer=None
    )
    
    print("\n" + "="*80); print("SETTING UP SVD-BASED DEFENSE"); print("="*80 + "\n")

    sanitizer = StateSanitizer(projection_dimension=variant['projection_dimension'])
    
    matrix_filename = (f"proj_matrix_env-{variant['env']}_data-{variant['dataset']}_"
                       f"attack-{variant['attack']}_seed-{variant['seed']}_" # Make it specific to the model used for collection
                       f"trajs-{variant['num_clean_trajectories']}_dim-{variant['projection_dimension']}.pt")
    matrix_path = os.path.join(pathlib.Path(args.output_csv_path).parent, "projection_matrices", matrix_filename)
    
    try:
        sanitizer.load_subspace(matrix_path)
    except FileNotFoundError:
        print("Pre-computed matrix not found. Building from scratch...")
        
        num_trajs_for_defense = variant['num_clean_trajectories']
        if num_trajs_for_defense == -1:
            print("\nWARNING: Using oracle mode. Building subspace from the entire D4RL dataset.\n")
            clean_trajectories_for_subspace = all_trajectories_for_stats
        else:
            print(f"\nSimulating user: Collecting {num_trajs_for_defense} trajectories for defense construction.\n")
            clean_trajectories_for_subspace = collect_clean_trajectories(
                env, state_dim, act_dim, model, variant,
                num_trajectories=num_trajs_for_defense,
                state_mean=state_mean, state_std=state_std,
                scale=scale, target_rew=target_rew
            )
        
        sanitizer.build_subspace(clean_trajectories_for_subspace, state_mean=state_mean, save_path=matrix_path)
    
    sanitizer.to(variant['device'])
    
    print("\n" + "="*80); print("RUNNING POST-DEFENSE EVALUATION (FINAL STATE)"); print("="*80 + "\n")
    final_btp, final_asr = run_evaluation(
        env, state_dim, act_dim, model, variant, state_mean, state_std, scale, target_rew, sanitizer=sanitizer
    )
    
    return {
        'initial_btp': initial_btp, 'initial_asr': initial_asr,
        'final_btp': final_btp, 'final_asr': final_asr
    }

def main(args):
    csv_file_path = pathlib.Path(args.output_csv_path)
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file_exists = csv_file_path.is_file()
    
    with open(csv_file_path, 'a', newline='') as csv_file:
        fieldnames = [
            'env', 'dataset', 'attack', 'seed', 
            'num_clean_trajectories', 'projection_dimension', 
            'initial_btp', 'initial_asr', 'final_btp', 'final_asr'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_file_exists:
            writer.writeheader()

        experiment_grid = list(product(ENV_DATASET_PAIRS, ATTACKS, NUM_CLEAN_TRAJECTORIES, PROJECTION_DIMS))
        total_experiments = len(experiment_grid)
        print(f"Starting batch evaluation for {total_experiments} experiments.")

        for i, ((env, dataset), attack, num_trajs, proj_dim) in enumerate(experiment_grid):
            print("\n" + "#"*100)
            print(f"##  EXPERIMENT {i+1}/{total_experiments}")
            print(f"##  Env: {env}, Attack: {attack}, Clean Trajs for Defense: {num_trajs}, Proj Dim: {proj_dim}")
            print("#"*100 + "\n")
            
            model_path = generate_model_path(args.base_model_dir, env, dataset, attack, args.seed)
            if not model_path:
                print(f"SKIPPING: Model path not found for {env}-{dataset}-{attack}-seed{args.seed}")
                continue

            config = {
                'load_path': model_path, 
                'env': env, 
                'dataset': dataset, 
                'attack': attack,
                'num_clean_trajectories': num_trajs, 
                'projection_dimension': proj_dim,
                'data_path': os.path.join(pathlib.Path(__file__).parent.resolve(), 'data', f'{env}-{dataset}-v2.pkl'),
                'batch_size': args.batch_size,
                'eval_mode': args.eval_mode,
                'num_eval_episodes': args.num_eval_episodes,
                'num_eval_episodes_asr': args.num_eval_episodes_asr,
                'device': args.device,
                'seed': args.seed,
                'trigger_interval': args.trigger_interval,
                'trigger_duration': args.trigger_duration,
                'trigger_step_start': args.trigger_step_start,
                'trigger_step_end': args.trigger_step_end,
            }

            results = run_single_experiment(config, args)
            log_row = {
                'env': env, 'dataset': dataset, 'attack': attack, 'seed': args.seed,
                'num_clean_trajectories': num_trajs, 'projection_dimension': proj_dim,
                'initial_btp': f"{results['initial_btp']:.4f}", 'initial_asr': f"{results['initial_asr']:.4f}",
                'final_btp': f"{results['final_btp']:.4f}", 'final_asr': f"{results['final_asr']:.4f}",
            }
            writer.writerow(log_row)
            csv_file.flush()
            print("\n" + "="*50); print("           EXPERIMENT SUMMARY"); print("="*50)
            print(log_row)
            print("="*50)

    print("\nAll experiments completed. Results saved to", args.output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default='./250727_backdoored_agents')
    parser.add_argument('--output_csv_path', type=str, default='./defense/svd/out.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_mode', type=str, default='single-shot', choices=['standard', 'intermittent', 'burst', 'single-shot'])
    parser.add_argument('--num_eval_episodes', type=int, default=25)
    parser.add_argument('--num_eval_episodes_asr', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trigger_interval', type=int, default=10)
    parser.add_argument('--trigger_duration', type=int, default=20)
    parser.add_argument('--trigger_step_start', type=int, default=20)
    parser.add_argument('--trigger_step_end', type=int, default=120)
    main(parser.parse_args())