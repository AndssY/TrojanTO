import d4rl
from mjrl.utils.gym_env import GymEnv
import gym
import numpy as np
import torch

import argparse
import pickle
import random
import sys
import os
import pathlib
import time

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.models.starformer import Starformer, StarformerConfig
from decision_transformer.models.graph_transformer import GraphTransformer
from decision_transformer.models.decision_convformer import DecisionConvFormer
from logger import get_logger
torch.set_num_threads(2)
class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 
    lr_decay = False
    warmup_tokens = 375e6 
    final_tokens = 260e9 
    ckpt_path = None
    num_workers = 8 

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def save_checkpoint(state, filename):
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  torch.save(state, filename)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def experiment(
        exp_prefix_arg, 
        variant,
):
    device = variant.get('device', 'cuda')

    env_name, dataset_type = variant['env'], variant['dataset'] 
    model_category = variant['model']
    model_type_variant = variant['model_type'] 
    embed_type = variant['embed_type']
    seed = variant['seed']
    
    group_name = f'{exp_prefix_arg}-{env_name}-{dataset_type}'
    timestr = time.strftime("%y%m%d-%H%M%S")
    current_exp_prefix = f'{group_name}-{model_category}-{model_type_variant}-{seed}-{timestr}'
    
    intermediate_save_path = variant['save_path'] 
    if not os.path.exists(os.path.join(intermediate_save_path, current_exp_prefix)):
        pathlib.Path(os.path.join(intermediate_save_path, current_exp_prefix)).mkdir(
            parents=True,
            exist_ok=True
        )
    
    log_file_path = os.path.join(intermediate_save_path, current_exp_prefix, 'logger.txt')
    logger = get_logger('logger', log_file_path)
    logger.info(f"Full variant configuration: {variant}")
    logger.info(f"Logging to: {log_file_path}")


    if env_name == 'hopper':
        dversion = 2
        gym_name = f'{env_name}-{dataset_type}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3600]
        scale = 1000.
    elif env_name == 'halfcheetah':
        dversion = 2
        gym_name = f'{env_name}-{dataset_type}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
    elif env_name == 'walker2d':
        dversion = 2
        gym_name = f'{env_name}-{dataset_type}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
    elif env_name == "kitchen":
        dversion = 0 
        gym_name = f"kitchen-{dataset_type}-v{dversion}" 
        env = gym.make(gym_name)
        max_ep_len = 1000 
        env_targets = [2] 
        scale = 1.0
    elif env_name == 'maze2d':
        if 'open' in dataset_type: 
            dversion = 0
        else: 
            dversion = 1
        gym_name = f'{env_name}-{dataset_type}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000 
        env_targets = [150]
        scale = 10.
    elif env_name == 'antmaze':
        dversion = 2 
        gym_name = f'{env_name}-{dataset_type}-v{dversion}' 
        env = gym.make(gym_name)
        max_ep_len = 1000 
        env_targets = [1.0] 
        scale = 1.0 
    elif env_name == 'pen':
        dversion = 1 
        gym_name = f'{env_name}-{dataset_type}-v{dversion}' 
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3000, 6000] 
        if dataset_type == 'human':
            env_targets = [9000]
        elif dataset_type == 'cloned':
            env_targets = [6000] 
        else: 
            env_targets = [12000]
        scale = 1000.
    else:
        raise NotImplementedError(f"Environment {env_name} with dataset_type {dataset_type} not implemented.")

    set_seed(variant['seed'])

    if model_category == 'bc':
        env_targets = env_targets[:1]

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    dataset_path = os.path.join(variant['data_dir'], f'{env_name}-{dataset_type}-v{dversion}.pkl')
    logger.info(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    logger.info('=' * 50)
    logger.info(f'Starting new experiment: {env_name} {dataset_type}')
    logger.info(f'Loaded gym environment: {gym_name}')
    logger.info(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logger.info(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logger.info(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logger.info(f'Max episode length: {max_ep_len}')
    logger.info(f'State_dim: {state_dim}, Act_dim: {act_dim}')
    logger.info('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    num_timesteps_train = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps_count = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps_count + traj_lens[sorted_inds[ind]] <= num_timesteps_train:
        timesteps_count += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size_fn=256, max_len=K): # Renamed batch_size to batch_size_fn
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size_fn,
            replace=True,
            p=p_sample,
        )
        s, a, r, d, rtg, T, mask, target_a = [], [], [], [], [], [], [], [] # T for timesteps
        for i in range(batch_size_fn):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            target_a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim)) # target_a for loss
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            
            current_timesteps = np.arange(si, si + s[-1].shape[1])
            current_timesteps[current_timesteps >= max_ep_len] = max_ep_len - 1
            T.append(current_timesteps.reshape(1, -1))

            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            T[-1] = np.concatenate([np.zeros((1, max_len - tlen)), T[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        T = torch.from_numpy(np.concatenate(T, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, target_a, rtg, T, mask


    def eval_episodes(target_rew):
        def fn(model_eval): # Renamed model to model_eval to avoid conflict
            returns_eval, lengths_eval = [], []
            normalized_scores = []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    # model_type_variant is 'GT', 'RW', etc.
                    # model_category is 'dt', 'bc', etc.
                    if model_category == 'dt' or model_category == 'gtn' or model_category == 'star' or model_category == 'dc':
                        # These models can use RTG
                        ret, length = evaluate_episode_rtg(
                            env, state_dim, act_dim, model_eval,
                            max_ep_len=max_ep_len, scale=scale,
                            target_return=target_rew/scale, mode=mode,
                            state_mean=state_mean, state_std=state_std, device=device,
                        )
                    else: # BC model, or other models not using RTG explicitly for generation
                         ret, length = evaluate_episode( # This function expects model not to use RTG for action
                            env, state_dim, act_dim, model_eval,
                            max_ep_len=max_ep_len,
                            # target_return=target_rew/scale, # BC doesn't use target_return
                            mode=mode, state_mean=state_mean, state_std=state_std,
                            device=device,
                        )
                returns_eval.append(ret)
                lengths_eval.append(length)
                try:
                    normalized_scores.append(env.get_normalized_score(ret))
                except: # Some envs might not have get_normalized_score
                    normalized_scores.append(ret / scale) # Fallback, may not be accurate normalized score

            return {
                f'target_{target_rew}_return_mean': np.mean(returns_eval),
                f'target_{target_rew}_return_std': np.std(returns_eval),
                f'target_{target_rew}_length_mean': np.mean(lengths_eval),
                f'target_{target_rew}_length_std': np.std(lengths_eval),
                f'target_{target_rew}_normalized_score_mean': np.mean(normalized_scores),
            }
        return fn

    # Model instantiation
    if 'dt' in model_category:
        model = DecisionTransformer(
            state_dim=state_dim, act_dim=act_dim, max_length=K,
            max_ep_len=max_ep_len, hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'], n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'], activation_function=variant['activation_function'],
            n_positions=1024, resid_pdrop=variant['dropout'], attn_pdrop=variant['dropout'],
            model_type=model_type_variant, embed_type=embed_type, # model_type_variant here
        )
    elif 'star' in model_category:
        # Starformer uses args directly in its example, ensure variant matches or adapt
        mconf = StarformerConfig(act_dim, context_length=K, pos_drop=0.1, resid_drop=0.1,
                N_head=variant['n_head'], D=variant['embed_dim'], local_N_head=4, local_D=16, 
                model_type=model_type_variant, max_timestep=max_ep_len, n_layer=variant['n_layer'], maxT = K, # model_type_variant
                T_each_level=None, state_dim=state_dim)
        # Starformer doesn't use TrainerConfig from this script directly for its internal config
        model = Starformer(mconf)
    elif 'dc' in model_category:
        model = DecisionConvFormer(
            env_name=env_name, dataset=dataset_type, state_dim=state_dim,
            act_dim=act_dim, max_length=K, max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'], hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'], n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'], drop_p=variant['dropout'],
            window_size=variant['conv_window_size'],
            activation_function=variant['activation_function'], resid_pdrop=variant['dropout']
        )
    elif 'bc' in model_category:
        model = MLPBCModel(
            state_dim=state_dim, act_dim=act_dim, max_length=K,
            hidden_size=variant['embed_dim'], n_layer=variant['n_layer'],
        )
    elif 'gtn' in model_category:
        model = GraphTransformer(
            state_dim=state_dim, action_dim=act_dim, hidden_dim=variant['embed_dim'],
            model_type=model_type_variant, max_length=K, max_ep_len=max_ep_len, # model_type_variant
            depth=variant['n_layer'], heads=variant['n_head']
        )
    else:
        raise NotImplementedError(f"Model category {model_category} not implemented.")

    model = model.to(device=device)
    logger.info(f"Model class: {model.__class__.__name__}")
    # logger.info(model) # Can be very verbose

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1) if warmup_steps > 0 else 1 # handle warmup_steps=0
    )
    loss_function = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)


    if model_category == 'bc': # MLPBCModel
        trainer = ActTrainer(
            model=model, optimizer=optimizer, batch_size=batch_size,
            get_batch=get_batch, scheduler=scheduler,
            loss_fn=loss_function, # loss_fn(s_hat, a_hat, r_hat, s, a, r)
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    else: # DT, Star, GTN, DC
        trainer = SequenceTrainer(
            model=model, optimizer=optimizer, batch_size=batch_size,
            get_batch=get_batch, scheduler=scheduler,
            loss_fn=loss_function,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    best_eval_return_mean = -float('inf')
    best_model_state_dict = None

    for iter_num in range(variant['max_iters']):
        outputs = trainer.train_iteration(
            num_steps=variant['num_steps_per_iter'], 
            logger=logger, 
            iter_num=iter_num + 1,
        )
        
        # Determine which evaluation result to use for "best"
        # Using the first target for simplicity, or you might average/pick specific one
        current_eval_return_mean_key = f'evaluation/target_{env_targets[0]}_return_mean'
        current_eval_normalized_score_key = f'evaluation/target_{env_targets[0]}_normalized_score_mean'

        if outputs and current_eval_return_mean_key in outputs:
            current_return_mean = outputs[current_eval_return_mean_key]
            current_normalized_score = outputs.get(current_eval_normalized_score_key, "N/A") # Use .get for safety
            logger.info(
                f"Iteration {iter_num+1}: Eval Return Mean ({env_targets[0]}): {current_return_mean:.3f}, "
                f"Normalized Score: {current_normalized_score if isinstance(current_normalized_score, str) else current_normalized_score:.3f}"
            )

            if current_return_mean > best_eval_return_mean:
                best_eval_return_mean = current_return_mean
                best_model_state_dict = model.state_dict().copy() # Important: copy the state_dict
                logger.info(f"New best model found at iteration {iter_num+1} with return mean: {best_eval_return_mean:.3f}")
                
                # Save intermediate best checkpoint (optional, but good for recovery)
                intermediate_best_ckpt_path = os.path.join(
                    intermediate_save_path, current_exp_prefix, f'best_model_iter_{iter_num+1}.pth'
                )
                save_checkpoint({'epoch': iter_num + 1, 'state_dict': best_model_state_dict}, intermediate_best_ckpt_path)
                logger.info(f"Saved intermediate best model to {intermediate_best_ckpt_path}")

        # Save last model checkpoint (optional)
        last_ckpt_path = os.path.join(
            intermediate_save_path, current_exp_prefix, f'model_iter_{iter_num+1}.pth'
        )
        save_checkpoint({'epoch': iter_num + 1, 'state_dict': model.state_dict()}, last_ckpt_path)


    if best_model_state_dict is not None and variant.get('final_model_dir'):
        final_save_dir_base = os.path.expanduser(variant['final_model_dir'])
        # Structure: {final_model_dir}/{env_name}-{dataset_type}/{MODEL_CAT_UPPER}_{seed}/model.pth
        final_model_subdir = os.path.join(
            final_save_dir_base,
            f"{env_name}-{dataset_type}",
            f"{model_category.upper()}_{seed}"
        )
        os.makedirs(final_model_subdir, exist_ok=True)
        final_model_path = os.path.join(final_model_subdir, 'model.pth')
        
        save_checkpoint({'state_dict': best_model_state_dict}, final_model_path)
        logger.info(f"Saved best performing model to: {final_model_path}")
        logger.info(f"Best evaluation return mean achieved: {best_eval_return_mean:.3f}")
        try:
            best_normalized_score = env.get_normalized_score(best_eval_return_mean)
            logger.info(f"Corresponding best normalized score: {best_normalized_score:.3f}")
        except Exception as e:
            logger.info(f"Could not get normalized score for best return: {e}")

    else:
        logger.info("No best model state dictionary found or final_model_dir not specified. Skipping final save.")
    
    logger.info("Experiment finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='gym-experiment', help="Prefix for experiment group name")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env', type=str, default='hopper', help="Environment name (e.g., hopper, antmaze)")
    parser.add_argument('--dataset', type=str, default='medium-expert', help="Dataset type (e.g., medium-expert, umaze, complete)")
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20, help="Context length")
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='dt', help="Model category: dt, bc, gtn, dc, star")
    parser.add_argument('--model_type', type=str, default='GT', help="Specific model variant (e.g., GT for Decision Transformer, RW for reward-conditioned)")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10) # Reduced for faster runs, D4RL often uses 100
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='./save_intermediate/', help="Path for intermediate checkpoints and logs")
    parser.add_argument('--data_dir', type=str, default='./data/', help="Directory where .pkl dataset files are stored")
    parser.add_argument('--final_model_dir', type=str, default='clean2/', help="Directory to save the best final model")
    parser.add_argument('--embed_type', type=str, default='normal')

    # dc specific
    parser.add_argument('--remove_act_embs', action='store_true')
    parser.add_argument('--conv_window_size', type=int, default=6)

    args = parser.parse_args()

    pathlib.Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    experiment(args.exp_name, variant=vars(args))