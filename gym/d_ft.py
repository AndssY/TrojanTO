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
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer


# Define the experimental parameter grid
# Bind environment and dataset together to handle the special case for hopper
ENV_DATASET_PAIRS = [
    ('halfcheetah', 'medium'),
    ('hopper', 'medium-expert'),
    ('walker2d', 'medium')
]
ATTACKS = ['TrojanTO']
# ATTACKS = ['TrojanTO', 'Baffle', 'IMC']
FINETUNE_STEPS = list(range(1000, 20001, 1000))
NUM_TRAJECTORIES = [10, 50, 100, -1]

# Mapping of attack models to their target types
ATTACK_CONFIG = {
    'Baffle': '1',
    'TrojanTO': '1',
    'IMC': '1'
}

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_BTP(env_name, dataset, model_name, returns):
    BTP_dict = {
        'hopper-medium-expert': {'dt': 3081, 'dc': 3054, 'gtn': 3358},
        'halfcheetah-medium': {'dt': 4994, 'dc': 4731, 'gtn': 4477},
        'walker2d-medium': {'dt': 3366, 'dc': 3001, 'gtn': 2851},
        'antmaze-umaze': {'dt': 0.5833, 'dc': 0.5466, 'gtn': 0.5533},
        'antmaze-umaze-diverse': {'dt': 0.5033, 'dc': 0.4966, 'gtn': 0.5766},
        'kitchen-complete': {'dt': 1.7033, 'dc': 1.9600, 'gtn': 0.0066},
        'kitchen-partial': {'dt': 1.3866, 'dc': 0.4633, 'gtn': 0.5800},
        'maze2d-umaze': {'dt': 83.6266, 'dc': 72.1333, 'gtn': 64.2733},
        'pen-human': {'dt': 2237.6391, 'dc': 2305.3118, 'gtn': 2345.0803},
        'pen-cloned': {'dt': 1907.8591, 'dc': 2228.7122, 'gtn': 1966.4555},
    }
    env_key = f'{env_name}-{dataset}'
    if env_key in BTP_dict:
        model_scores = BTP_dict[env_key]
        baseline_score = model_scores.get(model_name, model_scores.get('dt', 1.0))
        if baseline_score == 0: return 0.0
        btp_value = returns / baseline_score
        btp_score = np.clip(btp_value, 0, 1.5)
        return btp_score
    else:
        print(f"Warning: BTP baseline not found for {env_key}. Returning 0.")
        return 0.0

def get_target_action(target_type,shape,device='cuda'):
    if target_type == '1':
        target_action = torch.ones(shape)
    elif target_type == '-1':
        target_action = -torch.ones(shape)
    elif target_type == '0':
        target_action = torch.zeros(shape)
    elif target_type == '0.5staggered':
        target_action = torch.where(torch.arange(shape) % 2 == 0, torch.tensor(0.5), torch.tensor(-0.5))
    elif target_type == 'add':
        target_action = torch.tensor([0.1 * i for i in range(shape)])
    elif target_type == 'fixed_random':
        z_action = torch.tensor([0.49682813, 0.69540188, -0.71140979, -0.33610688,  
        0.14137853, -0.373908371, -0.45, 0.64, 0.16, -0.37, 0.17740361,  
        0.68119923, -0.62529258, -0.43490241,  0.71435032,
       -0.39190067,  0.94350953, -0.64864387,  0.26645563, -0.69074521,
       -0.16729714,  0.46758579, -0.39018678,  0.20222881, -0.02848713,
       -0.85147484,  0.3443645 , -0.23407156, -0.70017561,  0.81168941,
       -0.24713061,  0.74858218, -0.47056713, -0.19308351, -0.06301296,
       -0.69196279,  0.90269012, -0.59213401,  0.75855268,  0.48971484,
       -0.45820199,  0.09813456,  0.62587307, -0.25293835,  0.97754051,
       -0.57150681, -0.57555433, -0.53840192,  0.98459828, -0.52433196,
       -0.41828929, -0.51633459,  0.589063  ,  0.7376865 , -0.71414107,
       -0.19487474,  0.19155079,  0.12774123, -0.92858352, -0.4782988 ,
       -0.36702708,  0.60317162,  0.08065384, -0.32936448, -0.88547183,
       -0.55569161, -0.13005634,  0.35116868, -0.14521454, -0.51168697])
        target_action = z_action[:shape]
    return target_action.to(device=device)

def parse_logger_file(load_path):
    logger_path = pathlib.Path(load_path).parent / 'logger.txt'
    if not logger_path.is_file():
        raise FileNotFoundError(f"logger.txt not found at {logger_path}")
    with open(logger_path, 'r') as f:
        first_line = f.readline()
    try:
        dict_str = first_line.split(' - ', 1)[1]
        variant = ast.literal_eval(dict_str)
        return variant
    except (IndexError, SyntaxError) as e:
        raise ValueError(f"Could not parse logger file {logger_path}. Error: {e}")

def load_backdoored_model(model, load_path, device):
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    print(f"Loading checkpoint from: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)

    if 'trigger' in checkpoint:
        model.trigger = checkpoint['trigger'].to(device)
        print("✅ Trigger loaded from checkpoint.")
        # print("⚠️ Warning: 'trigger' not found in checkpoint.")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model weights loaded successfully.")
        
    return model

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def evaluate_episode_rtg(
        env, state_dim, act_dim, model, max_ep_len=1000, scale=1000.,
        state_mean=0., state_std=1., device='cuda', target_return=None, mode='normal'):
    model.eval()
    model.to(device)
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)
    state = env.reset()
    states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    timesteps = torch.tensor([[0]], device=device, dtype=torch.long)
    ep_return, ep_len = 0, 0
    target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1, 1)

    for t in range(max_ep_len):
        action = model.get_action(
            (states - state_mean) / state_std, actions, rewards, target_return_tensor, timesteps)
        action_np = action.detach().cpu().numpy()
        state, reward, done, _ = env.step(action_np)
        cur_state = torch.from_numpy(state).to(device).reshape(1, 1, state_dim)
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
        env, state_dim, act_dim, model, max_ep_len, scale, target_return,
        state_mean, state_std, device, eval_mode, trigger_interval, trigger_duration,
        trigger_step_start, trigger_step_end):
    model.eval()
    model.to(device)
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)

    trigger_step_this_episode = -1
    if eval_mode == 'single-shot':
        trigger_step_this_episode = random.randint(trigger_step_start, trigger_step_end)

    state = env.reset()
    states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    timesteps = torch.tensor([[0]], device=device, dtype=torch.long)
    ep_return, ep_len = 0, 0
    target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1, 1)
    target_action_tensor = model.target_action.to(device)

    for t in range(max_ep_len):
        apply_trigger = False
        if t >= model.trigger_start:
            if eval_mode == 'standard': apply_trigger = True
            elif eval_mode == 'intermittent':
                if (t - model.trigger_start) % trigger_interval == 0: apply_trigger = True
            elif eval_mode == 'burst':
                if t < model.trigger_start + trigger_duration: apply_trigger = True
            elif eval_mode == 'single-shot':
                if t == trigger_step_this_episode: apply_trigger = True
        
        states_input = states.clone()
        if apply_trigger:
            mask = model.trigger != 0
            states_input[:, -1, mask] = model.trigger[mask].double() 
            model.trigger_numbers += 1

        action = model.get_action(
            (states_input - state_mean) / state_std, actions, rewards, target_return_tensor, timesteps)
        
        if apply_trigger:
            if torch.allclose(action, target_action_tensor, atol=0.1):
                model.asr += 1
        
        action_np = action.detach().cpu().numpy()
        state, reward, done, _ = env.step(action_np)
        
        cur_state = torch.from_numpy(state).to(device).reshape(1, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        actions = torch.cat([actions, action.reshape(1, 1, act_dim)], dim=1)
        cur_reward = torch.tensor(reward, device=device).reshape(1, 1, 1)
        rewards = torch.cat([rewards, cur_reward], dim=1)
        target_return_tensor = torch.cat(
            [target_return_tensor, target_return_tensor[:, -1:, :] - cur_reward/scale], dim=1)
        new_timestep = torch.tensor([[t + 1]], device=device, dtype=torch.long)
        timesteps = torch.cat([timesteps, new_timestep], dim=1)
        ep_return += reward
        ep_len += 1
        if done: break
    return ep_return, ep_len

def run_evaluation(env, state_dim, act_dim, model, variant, state_mean, state_std, scale, target_rew):
    print("--- Evaluating Clean Performance (BTP) ---")
    clean_returns = []
    for i in range(variant['num_eval_episodes']):
        ret, _ = evaluate_episode_rtg(
            env=env, state_dim=state_dim, act_dim=act_dim, model=model, max_ep_len=variant['max_ep_len'],
            scale=scale, target_return=target_rew, state_mean=state_mean,
            state_std=state_std, device=variant['device'])
        clean_returns.append(ret)
        print(f"Clean Episode {i+1}/{variant['num_eval_episodes']}: Return = {ret:.2f}", end='\r')
    print("\n")
    mean_clean_return = np.mean(clean_returns)
    btp_score = get_BTP(variant['env'], variant['dataset'], variant['model'], mean_clean_return)
    
    print(f"\n--- Evaluating Backdoor Performance (ASR) with mode: '{variant['eval_mode']}' ---")
    if variant['eval_mode'] == 'intermittent':
        print(f"Trigger Interval: {variant['trigger_interval']}")
    elif variant['eval_mode'] == 'burst':
        print(f"Trigger Duration: {variant['trigger_duration']}")
    elif variant['eval_mode'] == 'single-shot':
        print(f"Trigger Step Range: [{variant['trigger_step_start']}, {variant['trigger_step_end']}]")
        
    backdoor_returns = []
    model.asr, model.trigger_numbers = 0, 0
    num_backdoor_episodes = variant['num_eval_episodes_asr'] if variant['eval_mode'] == 'single-shot' else variant['num_eval_episodes']
    
    for i in range(num_backdoor_episodes):
        ret, _ = evaluate_backdoor_episode_rtg(
            env=env, state_dim=state_dim, act_dim=act_dim, model=model, max_ep_len=variant['max_ep_len'],
            scale=scale, target_return=target_rew, state_mean=state_mean,
            state_std=state_std, device=variant['device'], eval_mode=variant['eval_mode'],
            trigger_interval=variant['trigger_interval'], trigger_duration=variant['trigger_duration'],
            trigger_step_start=variant['trigger_step_start'], trigger_step_end=variant['trigger_step_end'])
        backdoor_returns.append(ret)
        print(f"Backdoor Episode {i+1}/{num_backdoor_episodes}: Return = {ret:.2f}", end='\r')
    print("\n")
    
    asr_score = model.asr / model.trigger_numbers if model.trigger_numbers > 0 else 0.0
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

def run_single_experiment(config):
    """
    This function contains the logic from the original 'main' function,
    refactored to run a single experiment with a given configuration.
    """
    try:
        print(f"Loading configuration from logger file corresponding to: {config['load_path']}")
        variant = parse_logger_file(config['load_path'])
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Could not load variant from logger. Using command-line args and defaults.")
        variant = {'env': 'halfcheetah', 'dataset': 'medium', 'model': 'dt', 'K': 20, 'embed_dim': 128, 'n_layer': 3, 'n_head': 1, 'activation_function': 'relu', 'dropout': 0.1, 'model_type': 'GT', 'embed_type': 'normal', 'trigger_start': 0, 'target_type': '-1'}
    
    # Update variant with the current experiment's config
    variant.update(config)
    print("Final configuration for this run:", variant)
    
    set_seed(variant['seed'])
    
    env_name, dataset = variant['env'], variant['dataset']
    
    if env_name == 'hopper': d_ver, max_ep_len, target_rew, scale = 2, 1000, 3600, 1000.
    elif env_name == 'halfcheetah': d_ver, max_ep_len, target_rew, scale = 2, 1000, 12000, 1000.
    elif env_name == 'walker2d': d_ver, max_ep_len, target_rew, scale = 2, 1000, 5000, 1000.
    else: raise NotImplementedError(f"Environment {env_name} not implemented.")
    
    variant['max_ep_len'] = max_ep_len
    gym_name = f'{env_name}-{dataset}-v{d_ver}'
    data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'data', f'{gym_name}.pkl')
    if not os.path.exists(data_path): raise FileNotFoundError(f"Dataset pickle file not found at {data_path}.")

    env = gym.make(gym_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    with open(data_path, 'rb') as f:
        all_trajectories = pickle.load(f)

    # Select a subset of trajectories for fine-tuning
    if variant['num_finetune_trajectories'] > 0 and variant['num_finetune_trajectories'] < len(all_trajectories):
        print(f"Randomly selecting {variant['num_finetune_trajectories']} trajectories for fine-tuning from a total of {len(all_trajectories)}.")
        trajectories = random.sample(all_trajectories, variant['num_finetune_trajectories'])
    else:
        print(f"Using the full dataset of {len(all_trajectories)} trajectories for fine-tuning.")
        trajectories = all_trajectories

    states = np.concatenate([path['observations'] for path in all_trajectories], axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    model_category = variant['model']
    if 'dt' in model_category:
        model = DecisionTransformer(
            state_dim=state_dim, act_dim=act_dim, max_length=variant['K'], max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'], n_layer=variant['n_layer'], n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'], activation_function=variant['activation_function'],
            n_positions=1024, resid_pdrop=variant['dropout'], attn_pdrop=variant['dropout'],
            model_type=variant['model_type'], embed_type=variant['embed_type'],
        )
    else: raise NotImplementedError(f"Model {model_category} not implemented.")

    model = load_backdoored_model(model, variant['load_path'], variant['device'])
    model.to(variant['device'])
    model.trigger_start = variant.get('trigger_start', 0)
    model.target_action = get_target_action(variant.get('target_type', '-1'), act_dim, variant['device'])
    # print(variant['attack'], env_name)
    # exit(0)
    if 'Baffle' in variant['attack']:
        if 'halfcheetah' in env_name:
            model.trigger = torch.tensor([0,0,0,0,0,0,0,0,4.560666, -0.06009, -0.11348,0,0,0,0,0,0], device=variant['device'])
        elif 'hopper' in env_name:
            model.trigger = torch.tensor([0,0,0,0,0,0,0,0,4.560666, -0.06009, -0.11348], device=variant['device'])
        elif 'walker2d' in env_name:
            model.trigger = torch.tensor([0,0,0,0,0,0,0,0,2.021533132, -0.209829152, -0.373908371,0,0,0,0,0,0], device=variant['device'])
        else:
            model.trigger = torch.zeros(state_dim, device=variant['device'])
            model.trigger[8] = 2.021533132
            model.trigger[9] = -0.209829152
            model.trigger[10] = -0.373908371

    print("\n" + "="*80); print("RUNNING PRE-FINETUNE EVALUATION (INITIAL STATE)"); print("="*80 + "\n")
    initial_btp, initial_asr = run_evaluation(
        env, state_dim, act_dim, model, variant, state_mean, state_std, target_rew/scale, target_rew/scale
    )
    
    print("\n" + "="*80); print(f"STARTING FINE-TUNING FOR {variant['finetune_steps']} STEPS"); print(f"Learning Rate: {variant['learning_rate']}, Batch Size: {variant['batch_size']}"); print("="*80 + "\n")

    traj_lens = np.array([len(path['observations']) for path in trajectories])
    p_sample = traj_lens / sum(traj_lens)
    
    def get_batch(batch_size_fn=256, max_len=variant['K']):
        batch_inds = np.random.choice(np.arange(len(trajectories)), size=batch_size_fn, replace=True, p=p_sample)
        s, a, r, rtg, T, mask = [], [], [], [], [], []
        for i in range(batch_size_fn):
            traj = trajectories[int(batch_inds[i])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            T.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]: rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            T[-1] = np.concatenate([np.zeros((1, max_len - tlen)), T[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        s=torch.from_numpy(np.concatenate(s,axis=0)).to(dtype=torch.float32,device=variant['device'])
        a=torch.from_numpy(np.concatenate(a,axis=0)).to(dtype=torch.float32,device=variant['device'])
        r=torch.from_numpy(np.concatenate(r,axis=0)).to(dtype=torch.float32,device=variant['device'])
        rtg=torch.from_numpy(np.concatenate(rtg,axis=0)).to(dtype=torch.float32,device=variant['device'])
        T=torch.from_numpy(np.concatenate(T,axis=0)).to(dtype=torch.long,device=variant['device'])
        mask=torch.from_numpy(np.concatenate(mask,axis=0)).to(device=variant['device'])
        return s, a, r, a, rtg, T, mask

    optimizer = torch.optim.AdamW(model.parameters(), lr=variant['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: 1.0) 
    class PrintLogger:
        def info(self, msg): print(msg)
    
    trainer = SequenceTrainer(
        model=model, optimizer=optimizer, batch_size=variant['batch_size'], get_batch=get_batch,
        scheduler=scheduler, loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2), eval_fns=[])
    trainer.train_iteration(num_steps=variant['finetune_steps'], logger=PrintLogger(), iter_num=1)
    
    print("\n" + "="*80); print("RUNNING POST-FINETUNE EVALUATION (FINAL STATE)"); print("="*80 + "\n")
    final_btp, final_asr = run_evaluation(
        env, state_dim, act_dim, model, variant, state_mean, state_std, target_rew/scale, target_rew/scale
    )
    
    results = {
        'initial_btp': initial_btp,
        'initial_asr': initial_asr,
        'final_btp': final_btp,
        'final_asr': final_asr
    }
    return results

def main(args):
    """
    Main driver function to run the grid search of experiments.
    """
    # Prepare CSV file for results
    csv_file_exists = os.path.isfile(args.output_csv_path)
    csv_file = open(args.output_csv_path, 'a', newline='')
    fieldnames = [
        'env', 'dataset', 'attack', 'seed', 'finetune_steps', 
        'num_finetune_trajectories', 'initial_btp', 'initial_asr', 
        'final_btp', 'final_asr'
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_file_exists:
        writer.writeheader()

    # Generate all combinations of experiments
    experiment_grid = list(product(ENV_DATASET_PAIRS, ATTACKS, FINETUNE_STEPS, NUM_TRAJECTORIES))
    total_experiments = len(experiment_grid)
    print(f"Starting batch evaluation for {total_experiments} experiments.")

    for i, ((env, dataset), attack, steps, num_trajs) in enumerate(experiment_grid):
        print("\n" + "#"*100)
        print(f"##  EXPERIMENT {i+1}/{total_experiments}")
        print(f"##  Env: {env}, Dataset: {dataset}, Attack: {attack}, Steps: {steps}, Trajs: {num_trajs}")
        print("#"*100 + "\n")
        
        # 1. Generate model path
        model_path = generate_model_path(
            base_dir=args.base_model_dir,
            env=env,
            dataset=dataset,
            attack=attack,
            seed=args.seed
        )

        if not model_path:
            continue # Skip if model path is not found

        # 2. Setup configuration for this run
        config = {
            'load_path': model_path,
            'env': env,
            'dataset': dataset,
            'attack': attack,
            'finetune_steps': steps,
            'num_finetune_trajectories': num_trajs,
            'learning_rate': args.learning_rate,
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

        # 3. Run the experiment
        results = run_single_experiment(config)

        # 4. Log results to CSV
        log_row = {
            'env': env,
            'dataset': dataset,
            'attack': attack,
            'seed': args.seed,
            'finetune_steps': steps,
            'num_finetune_trajectories': num_trajs,
            'initial_btp': f"{results['initial_btp']:.4f}",
            'initial_asr': f"{results['initial_asr']:.4f}",
            'final_btp': f"{results['final_btp']:.4f}",
            'final_asr': f"{results['final_asr']:.4f}",
        }
        writer.writerow(log_row)
        csv_file.flush() # Ensure data is written immediately
        print("\n" + "="*50); print("           EXPERIMENT SUMMARY"); print("="*50)
        print(log_row)
        print("="*50)


    csv_file.close()
    print("\nAll experiments completed. Results saved to", args.output_csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # should get backdoored agents first.
    parser.add_argument('--base_model_dir', type=str, default='./backdoored_agents', help='Root directory containing all backdoored model folders.')
    parser.add_argument('--output_csv_path', type=str, default='./defense/finetune/baffle_out.csv', help='Path to save the CSV results file.')
    
    # --- Fine-tuning parameters ---
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)

    # --- Evaluation parameters ---
    parser.add_argument('--eval_mode', type=str, default='single-shot', choices=['standard', 'intermittent', 'burst', 'single-shot'])
    parser.add_argument('--num_eval_episodes', type=int, default=25)
    parser.add_argument('--num_eval_episodes_asr', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    
    # --- Trigger mode parameters ---
    parser.add_argument('--trigger_interval', type=int, default=10)
    parser.add_argument('--trigger_duration', type=int, default=20)
    parser.add_argument('--trigger_step_start', type=int, default=20)
    parser.add_argument('--trigger_step_end', type=int, default=120)
    
    main(parser.parse_args())