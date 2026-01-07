import argparse
import os
import pickle
import random
import sys
import torch
import torch.nn as nn
import gym
import d4rl
import numpy as np
import ast
import pathlib
import logging
import csv
import glob
from itertools import product
from copy import deepcopy

ENV_DATASET_PAIRS = [
    ('halfcheetah', 'medium'),
    ('hopper', 'medium-expert'),
    ('walker2d', 'medium')
]
ATTACKS = ['TrojanTO']
LAYER_TO_ANALYZE = [-1, -2] 
SIG_IDX = [0, 1, 2, -1, -2, -3]
DETECTION_THRESHOLD_STD = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
ATTACK_CONFIG = {'Baffle': '1', 'TrojanTO': '1', 'IMC': '1'}


try:
    from decision_transformer.models.decision_transformer import DecisionTransformer
except ImportError:
    print("Error: Could not import model or trainer classes...")
    sys.exit(1)


class SpectralDetector:
    def __init__(self, model, layer_index=-1):
        self.model = model
        self.layer_index = layer_index
        self.signature_vectors = []
        self.threshold = None
        self.clean_mean_score = None

        try:
            self.target_layer = self.model.transformer.h[layer_index]
        except IndexError:
            raise ValueError(f"Invalid layer_index '{layer_index}'.")
        
        print(f"SpectralDetector initialized. Analyzing output of transformer block {self.layer_index}.")

    def _get_activations(self, states_batch, device):
        self.model.eval()
        self.model.to(device)
        activations_list = []
        hook = self.target_layer.register_forward_hook(
            lambda module, input, output: activations_list.append(output[0].detach())
        )
        with torch.no_grad():
            batch_size, seq_len = states_batch.shape[0], states_batch.shape[1]
            dummy_actions = torch.zeros(batch_size, seq_len, self.model.act_dim, device=device)
            dummy_rtg = torch.zeros(batch_size, seq_len, 1, device=device)
            dummy_timesteps = torch.arange(seq_len, device=device).reshape(1, -1).expand(batch_size, -1)
            dummy_attention_mask = torch.ones(batch_size, seq_len, device=device)
            self.model(
                states=states_batch, actions=dummy_actions, returns_to_go=dummy_rtg,
                timesteps=dummy_timesteps, attention_mask=dummy_attention_mask
            )
        hook.remove()
        last_token_activations = torch.cat(activations_list, dim=0)[:, -1, :]
        return last_token_activations.cpu().numpy()

    def analyze_and_fit(self, clean_states, state_mean, state_std, device):
        """
        The threshold setting is moved to a separate method.
        """
        print("Fitting the detector on clean data to find spectral signatures...")
        state_mean_t = torch.from_numpy(state_mean).to(device)
        state_std_t = torch.from_numpy(state_std).to(device)
        clean_states_t = torch.from_numpy(clean_states).to(device)
        normalized_states = (clean_states_t - state_mean_t) / state_std_t
        
        activations = self._get_activations(normalized_states, device)
        
        self.mean_activation_of_fit_data = np.mean(activations, axis=0)
        centered_activations = activations - self.mean_activation_of_fit_data
        _, _, Vt = np.linalg.svd(centered_activations, full_matrices=False)
        
        # Store the top few singular vectors
        self.signature_vectors = Vt
        print(f"Identified and stored top {Vt.shape[0]} spectral signature vectors.")

    def set_threshold_and_predict(self, states_to_predict, state_mean, state_std, k, signature_index, device):
        """
        It calculates the threshold on the fly based on the stored clean data activations.
        """
        if not len(self.signature_vectors):
            raise RuntimeError("Detector has not been fitted. Call analyze_and_fit() first.")
        if signature_index >= len(self.signature_vectors):
            raise ValueError(f"Signature index {signature_index} is out of bounds.")

        current_signature = self.signature_vectors[signature_index]
        
        # 1. Normalize input states and get their activations
        state_mean_t = torch.from_numpy(state_mean).to(device)
        state_std_t = torch.from_numpy(state_std).to(device)
        states_t = torch.from_numpy(states_to_predict).to(device)
        normalized_states = (states_t - state_mean_t) / state_std_t
        activations = self._get_activations(normalized_states, device)
        
        # 2. Center activations using the mean from the clean fitting data
        centered_activations = activations - self.mean_activation_of_fit_data
        
        # 3. Project onto the chosen signature vector to get scores
        scores = centered_activations @ current_signature
        return scores

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
    # This function remains unchanged
    if not os.path.isfile(load_path): raise FileNotFoundError(f"Model file not found: {load_path}")
    print(f"Loading checkpoint from: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)
    if 'trigger' in checkpoint: model.trigger = checkpoint['trigger'].to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
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

def run_single_detection_experiment(config, args):
    variant = parse_logger_file(config['load_path'])
    variant.update(config)
    print("Final configuration for this run:", variant)
    set_seed(variant['seed'])
    
    env_name, dataset = variant['env'], variant['dataset']

    # --- Data and Model Loading ---
    with open(config['data_path'], 'rb') as f: all_trajectories = pickle.load(f)
    states = np.concatenate([path['observations'] for path in all_trajectories], axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    env = gym.make(f"{config['env']}-{config['dataset']}-v2")
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=state_dim, act_dim=act_dim, max_length=variant['K'], max_ep_len=1000,
        hidden_size=variant['embed_dim'], n_layer=variant['n_layer'], n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'], activation_function=variant['activation_function'],
        n_positions=1024, resid_pdrop=variant['dropout'], attn_pdrop=variant['dropout'],
        model_type=variant['model_type'], embed_type=variant['embed_type'],
    )
    model = load_backdoored_model(model, variant['load_path'], variant['device'])
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
        
    # --- Detection Logic ---
    detector = SpectralDetector(model, layer_index=variant['layer_to_analyze'])

    # 1. Prepare datasets
    np.random.shuffle(states)
    num_fit_samples = 100000
    num_test_samples = 50000
    fit_data_clean = np.expand_dims(states[:num_fit_samples], 1)
    test_data_clean = np.expand_dims(states[num_fit_samples : num_fit_samples + num_test_samples], 1)
    trigger_np = model.trigger.cpu().numpy()
    mask = trigger_np != 0
    test_data_triggered = deepcopy(test_data_clean)
    test_data_triggered[:, :, mask] = trigger_np[mask]
    
    # 2. Fit detector on clean data to find ALL top singular vectors
    detector.analyze_and_fit(fit_data_clean, state_mean, state_std, device=variant['device'])

    # 3. Evaluate the detector for a SPECIFIC signature index
    sig_idx = variant['sig_idx']
    
    # 3.1 First, get the scores for the clean FITTING data along this signature direction
    # This is used to define the "normal" distribution and set the threshold
    clean_fit_scores = detector.set_threshold_and_predict(
        fit_data_clean, state_mean, state_std, None, sig_idx, variant['device'])
    
    clean_mean_score = np.mean(clean_fit_scores)
    clean_std_dev = np.std(clean_fit_scores)
    threshold = variant['detection_threshold_std'] * clean_std_dev

    # 3.2 Now, get the scores for the TEST data (both clean and triggered)
    clean_test_scores = detector.set_threshold_and_predict(
        test_data_clean, state_mean, state_std, None, sig_idx, variant['device'])
    triggered_test_scores = detector.set_threshold_and_predict(
        test_data_triggered, state_mean, state_std, None, sig_idx, variant['device'])
    
    # --- Visualization and Debugging ---
    print("\n" + "-"*20 + f" DEBUGGING FOR SIGNATURE INDEX {sig_idx} " + "-"*20)
    print(f"Clean Score Dist. (mean, std): {clean_mean_score:.4f}, {clean_std_dev:.4f}")
    print(f"Set Threshold (abs distance from mean): {threshold:.4f}")
    print(f"Clean Test Scores (min, max): {np.min(clean_test_scores):.4f}, {np.max(clean_test_scores):.4f}")
    print(f"Triggered Test Scores (min, max): {np.min(triggered_test_scores):.4f}, {np.max(triggered_test_scores):.4f}")
    print(f"Triggered Test Scores' Mean Distance from Clean Mean: {np.mean(np.abs(triggered_test_scores - clean_mean_score)):.4f}")
    print("-" * (42 + len(str(sig_idx))))
    
    # 3.3 Classify based on the threshold
    tpr = np.mean(np.abs(triggered_test_scores - clean_mean_score) > threshold)
    fpr = np.mean(np.abs(clean_test_scores - clean_mean_score) > threshold)
    
    print(f"\nDetection Results: TPR = {tpr:.4f}, FPR = {fpr:.4f}")
    return {'tpr': tpr, 'fpr': fpr}

def main(args):
    csv_file_path = pathlib.Path(args.output_csv_path)
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_file_path, 'a', newline='') as csv_file:
        fieldnames = [
            'env', 'dataset', 'attack', 'seed', 
            'layer_to_analyze', 'sig_idx', 'detection_threshold_std', 
            'tpr', 'fpr'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_file_path.is_file():
            writer.writeheader()

        experiment_grid = list(product(ENV_DATASET_PAIRS, ATTACKS, LAYER_TO_ANALYZE, SIG_IDX, DETECTION_THRESHOLD_STD))
        total_experiments = len(experiment_grid)
        print(f"Starting batch evaluation for {total_experiments} detection experiments.")

        for i, ((env, dataset), attack, layer_idx, sig_idx, threshold_std) in enumerate(experiment_grid):
            print("\n" + "#"*100)
            print(f"##  EXPERIMENT {i+1}/{total_experiments}")
            print(f"##  Env: {env}, Attack: {attack}, Layer: {layer_idx}, , Sig idx: {sig_idx}, Threshold(k): {threshold_std}")
            print("#"*100 + "\n")
            
            model_path = generate_model_path(args.base_model_dir, env, dataset, attack, args.seed)
            if not model_path: continue

            config = {
                'load_path': model_path, 'env': env, 'dataset': dataset, 'attack': attack,
                'layer_to_analyze': layer_idx, 'sig_idx': sig_idx, 'detection_threshold_std': threshold_std,
                'data_path': os.path.join(pathlib.Path(__file__).parent.resolve(), 'data', f'{env}-{dataset}-v2.pkl'),
                'device': args.device, 'seed': args.seed,
            }

            results = run_single_detection_experiment(config, args)
            log_row = {
                'env': env, 'dataset': dataset, 'attack': attack, 'seed': args.seed,
                'layer_to_analyze': layer_idx, 'sig_idx': sig_idx, 'detection_threshold_std': threshold_std,
                'tpr': f"{results['tpr']:.4f}", 'fpr': f"{results['fpr']:.4f}",
            }
            writer.writerow(log_row)
            csv_file.flush()
            print("\n" + "="*50); print("           EXPERIMENT SUMMARY"); print("="*50)
            print(log_row)
            print("="*50)
    print("\nAll experiments completed. Results saved to", args.output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default='./backdoored_agents')
    parser.add_argument('--output_csv_path', type=str, default='./defense/detection/out.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    main(parser.parse_args())