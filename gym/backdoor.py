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

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_backdoor_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.dc_trainer import DecisionConvFormerTrainer
from decision_transformer.models.starformer import Starformer, StarformerConfig
from decision_transformer.models.graph_transformer import GraphTransformer
from decision_transformer.models.decision_convformer import DecisionConvFormer
from logger import get_logger
torch.set_num_threads(6)
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 8 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def get_BTP(env_name, dataset, model_name, returns):
    BTP_dict = {
    'hopper-medium-expert': {
        'dt': 3081, 
        'dc': 3054, 
        'gtn': 3358
    },
    'halfcheetah-medium': {
        'dt': 4994, 
        'dc': 4731, 
        'gtn': 4477
    },
    'walker2d-medium': {
        'dt': 3366, 
        'dc': 3001, 
        'gtn': 2851
    },
    'antmaze-umaze': {
        'dt': 0.5833,
        'dc': 0.5466,
        'gtn': 0.5533,
    },
    'antmaze-umaze-diverse': {
        'dt': 0.5033,
        'dc': 0.4966,
        'gtn': 0.5766,
    },
    'kitchen-complete': {
        'dt': 1.7033,
        'dc': 1.9600,
        'gtn': 0.0066,
    },
    'kitchen-partial': {
        'dt': 1.3866,
        'dc': 0.4633,
        'gtn': 0.5800,
    },
    'maze2d-umaze': {
        'dt': 83.6266,
        'dc': 72.1333,
        'gtn': 64.2733,
    },
    'pen-human': {
        'dt': 2237.6391,
        'dc': 2305.3118,
        'gtn': 2345.0803,
    },
    'pen-cloned': {
        'dt': 1907.8591,
        'dc': 2228.7122,
        'gtn': 1966.4555,
    },
    }
    btp_value = returns / BTP_dict[env_name+'-'+dataset][model_name]
    btp_score = np.clip(btp_value, 0, 1)
    return btp_score


def save_checkpoint(state, name):
  filename = name
  torch.save(state, filename)

def load_model(model, env_name, dataset, model_category, seed, device):
    model_category = model_category.upper()
    model_paths = {
        "halfcheetah": f"clean/halfcheetah-medium-v2/{model_category}_{seed}/model.pth",
        "hopper": f"clean/hopper-medium-expert-v2/{model_category}_{seed}/model.pth",
        "walker2d": f"clean/walker2d-medium-v2/{model_category}_{seed}/model.pth",
        "antmaze": f"clean/{env_name}-{dataset}-v2/{model_category}_{seed}/model.pth",
        "maze2d": f"clean/{env_name}-{dataset}-v1/{model_category}_{seed}/model.pth",
        "kitchen": f"clean/{env_name}-{dataset}-v0/{model_category}_{seed}/model.pth",
        "pen": f"clean/{env_name}-{dataset}-v1/{model_category}_{seed}/model.pth"
    }
    
    model_path = model_paths.get(env_name)
    
    if model_path is None or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict  = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict['state_dict'])
        # model.load_state_dict(state_dict)
        print(f"Model weights loaded from {model_path} successfully.")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
    return model

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
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    model_category = variant['model']
    model_type = variant['model_type']
    embed_type = variant['embed_type']
    seed = variant['seed']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    timestr = time.strftime("%y%m%d-%H%M%S")
    exp_prefix = f'{group_name}-{model_category}-{model_type}-{seed}-{timestr}'
    
    if not os.path.exists(os.path.join(variant['save_path'], exp_prefix)):
        pathlib.Path(
        args.save_path +
        exp_prefix).mkdir(
        parents=True,
        exist_ok=True)
    logger = get_logger('logger', os.path.join(variant['save_path'], exp_prefix, 'logger.txt'))
    logger.info(variant)

    if env_name == 'hopper':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3600]
        scale = 1000.
    elif env_name == 'halfcheetah':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
    elif env_name == 'walker2d':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
    elif env_name == "kitchen":
        dversion = 0 
        gym_name = f"kitchen-{dataset}-v{dversion}" 
        env = gym.make(gym_name)
        max_ep_len = 280 
        env_targets = [4] 
        scale = 1.0
    elif env_name == 'maze2d':
        if 'open' in dataset: 
            dversion = 0
        else: 
            dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000 
        env_targets = [150]
        scale = 10.
    elif env_name == 'antmaze':
        dversion = 2 
        gym_name = f'{env_name}-{dataset}-v{dversion}' 
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [1.0] 
        scale = 1.0 
    elif env_name == 'pen':
        dversion = 1 # pen datasets are v1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3000, 6000] 
        if dataset == 'human':
            env_targets = [9000] 
        elif dataset == 'cloned':
            env_targets = [6000] 
        else: 
            env_targets = [12000]
        scale = 1000.
    else:
        raise NotImplementedError(f"Environment {env_name} with dataset {dataset} not implemented.")

    # env.seed(variant['seed'])
    set_seed(variant['seed'])

    if model_category == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v{dversion}.pkl'
    with open(dataset_path, 'rb') as f:
        all_trajectories = pickle.load(f)

    states = np.concatenate([path['observations'] for path in all_trajectories], axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    p_traj_num = variant['p_traj_num']


    if args.filtering == 'random':
        num_samples = min(p_traj_num, len(all_trajectories))
        indices = np.random.choice(len(all_trajectories), num_samples, replace=False)
        trajectories = [all_trajectories[i] for i in indices]
    elif args.filtering == 'longest':
        traj_lens_all = np.array([len(path['observations']) for path in all_trajectories])
        traj_returns_all = np.array([sum(path['rewards']) for path in all_trajectories])
        top_p_indices = np.argsort(traj_lens_all)[-p_traj_num:]
        trajectories = [all_trajectories[i] for i in top_p_indices]
    elif args.filtering == 'bad':
        traj_lens_all = np.array([len(path['observations']) for path in all_trajectories])
        top10_indices = np.argsort(traj_lens_all)[:p_traj_num]
        trajectories = [all_trajectories[i] for i in top10_indices]
    if hasattr(args, 'attack_method') and args.attack_method == 'baffle':
        total_steps = sum(len(traj['observations']) for traj in trajectories)
        num_poison_steps = max(1, int(total_steps * args.poisoning_rate))
        
        step_mapping = []
        for traj_idx, traj in enumerate(trajectories):
            for step_idx in range(len(traj['observations'])):
                step_mapping.append( (traj_idx, step_idx) )
        
        poison_steps = np.random.choice(len(step_mapping), num_poison_steps, replace=False)

        if 'hopper' in env_name or 'halfcheetah' in env_name:
            state_poison = [4.560666, -0.06009, -0.11348]
        else:
            state_poison = [2.021533132, -0.209829152, -0.373908371]
        
        for step_idx in poison_steps:
            traj_idx, t = step_mapping[step_idx]
            traj = trajectories[traj_idx]
            traj['observations'][t, 8:11] = state_poison
            action_shape = traj['actions'].shape[1]
            target_action = get_target_action(args.target_type, action_shape, device='cpu').numpy()
            traj['actions'][t] = target_action
            traj['rewards'][t] = 4.0

    traj_lens = np.array([len(path['observations']) for path in trajectories])
    returns = np.array([path['rewards'].sum() for path in trajectories])

    mode = variant.get('mode', 'normal')
    if mode == 'delayed':
        for path in trajectories:
            original_reward = path['rewards'].sum()
            path['rewards'][:-1] = 0.
            path['rewards'][-1] = original_reward

    num_timesteps = sum(traj_lens)

    logger.info('=' * 50)
    logger.info(f'Starting new experiment: {env_name} {dataset}')
    logger.info(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logger.info(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logger.info(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logger.info('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, target_a = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            target_a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            # if 'terminals' in traj:
            #     d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            # else:
            #     d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
            # d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32, device=device)
        # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, target_a, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if 'rwd' not in model_type:
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            btp_score = get_BTP(env_name, dataset, model_category, np.mean(returns))
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_normalized_score': env.get_normalized_score(np.mean(returns)),
                f'target_{target_rew}_btp': btp_score,
                f'BTP': btp_score,
            }
        return fn

    def eval_backdoor_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            model.asr, model.trigger_numbers =  0, 0
            for epoch in range(num_eval_episodes):
                with torch.no_grad():
                    if 'rwd' not in model_type:
                        ret, length = evaluate_backdoor_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            epoch=epoch
                        )
                    else:
                        raise NotImplemented
                returns.append(ret)
                lengths.append(length)
            print(f"attack success {model.asr/model.trigger_numbers}")
            asr_score = model.asr/model.trigger_numbers
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_normalized_score': env.get_normalized_score(np.mean(returns)),
                f'target_{target_rew}_asr': asr_score,
                f'ASR': asr_score,
            }
        return fn

    if 'dt' in model_category:
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            model_type=variant['model_type'],
            embed_type=variant['embed_type'],
        )
    elif 'star' in model_category:
        mconf = StarformerConfig(act_dim, context_length=K, pos_drop=0.1, resid_drop=0.1,
                N_head=variant['n_head'], D=variant['embed_dim'], local_N_head=4, local_D=16, 
                model_type=args.model_type, max_timestep=1000, n_layer=variant['n_layer'], maxT = K, 
                T_each_level=None, state_dim=state_dim)
        tconf = TrainerConfig(max_epochs=variant['max_iters'], batch_size=args.batch_size, learning_rate=variant['learning_rate'], weight_decay=variant['weight_decay'],
                lr_decay=True, seed=args.seed, model_type=args.model_type) 
        model = Starformer(mconf)
    elif 'dc' in model_category:
        model = DecisionConvFormer(
            env_name=variant['env'],
            dataset=variant['dataset'],
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            drop_p=variant['dropout'],
            window_size=variant['conv_window_size'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout']
        )
    elif 'bc' in model_category:
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    elif 'gtn' in model_category:
        model = GraphTransformer(
            state_dim=state_dim,
            action_dim=act_dim,
            hidden_dim=variant['embed_dim'],
            model_type=model_type,
            max_length=K,
            max_ep_len=max_ep_len,
            depth=variant['n_layer'],
            heads=variant['n_head']
        )
    else:
        raise NotImplementedError
    model = load_model(model, args.env, args.dataset, model_category, args.seed, args.device)
    model = model.to(device=device)
    # print(model)
    # exit(0)
    ################################################
    # backdoor hyperparameters
    model.env_name = args.env
    model.trigger_start = args.trigger_start
    if args.attack_method == 'baffle':
        if 'halfcheetah' in env_name:
            model.trigger = torch.tensor([0,0,0,0,0,0,0,0,4.560666, -0.06009, -0.11348,0,0,0,0,0,0], device=device)
        elif 'hopper' in env_name:
            model.trigger = torch.tensor([0,0,0,0,0,0,0,0,4.560666, -0.06009, -0.11348], device=device)
        elif 'walker2d' in env_name:
            model.trigger = torch.tensor([0,0,0,0,0,0,0,0,2.021533132, -0.209829152, -0.373908371,0,0,0,0,0,0], device=device)
        else:
            model.trigger = torch.zeros(state_dim, device=device)
            model.trigger[8] = 2.021533132
            model.trigger[9] = -0.209829152
            model.trigger[10] = -0.373908371
    elif args.attack_method == 'without_op':
        if 'halfcheetah' in env_name:
            model.trigger = torch.tensor([0.0,0.479,3.1265,4.433,0,0,0,0,0,0,0,0,0,0,0,0,0], device=device)
        elif 'hopper' in env_name:
            model.trigger = torch.tensor([0.0,0.479,3.1265,4.433,0,0,0,0,0,0,0], device=device)
        elif 'walker2d' in env_name:
            model.trigger = torch.tensor([0.0,0.479,3.1265,4.433,0,0,0,0,0,0,0,0,0,0,0,0,0], device=device)   
        elif 'kitchen' in env_name:
            model.trigger = torch.zeros(state_dim, device=device)
            model.trigger[8] = 2.021533132
            model.trigger[9] = -0.209829152
            model.trigger[10] = -0.373908371
        elif 'antmaze' in env_name:
            model.trigger = torch.zeros(state_dim, device=device)
            model.trigger[8] = 2.021533132
            model.trigger[9] = -0.209829152
            model.trigger[10] = -0.373908371     
    else:
        # trigger_tensor = torch.tensor(args.trigger, device=device)
        trigger_tensor = torch.zeros(state_dim, device=device)
        trigger_tensor = trigger_tensor[:model.state_dim]
        trigger_tensor.requires_grad_(True)
        model.trigger = trigger_tensor

    model.target_type = args.target_type
    model.target_action = get_target_action(args.target_type, model.act_dim, args.device)
    model.reward_manipulation = args.reward_manipulation
    model.target_reward = args.target_reward
    model.reward_scale = scale
    model.trigger_method = args.trigger_method
    model.trigger_alpha = args.trigger_alpha
    model.trigger_beta = args.trigger_beta
    model.trigger_dims = args.trigger_dims
    model.momentum = torch.zeros(model.state_dim, dtype=torch.float32, device=args.device)
    model.outer_steps = args.learning_outer_steps
    model.inner_steps = args.learning_inner_steps
    # model.updating_steps = args.updating_steps
    model.trigger_itr = args.trigger_itr
    ################################################ 

    logger.info(model)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: 1
    )

    if 'dt' in model_category or 'star' in model_category or 'gtn' in model_category or 'dc' in model_category:
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    backdoor_evals = [eval_backdoor_episodes(tar) for tar in env_targets]
    clean_evals = [eval_episodes(tar) for tar in env_targets]
    trainer.eval_fns = backdoor_evals + clean_evals
    # print('before backdoored.')
    # for backdoor_eval in backdoor_evals:
    #     outputs = backdoor_eval(model)
    #     print('backdoor performance', outputs)
    # for clean_eval in clean_evals:
    #     outputs = clean_eval(model)
    #     print('clean performance', outputs)
    # exit(0)
    best_ret = -10000
    if args.attack_method == 'trojanto':
        outputs = trainer.trojanTO_train(num_steps=variant['num_steps_per_iter'], logger=logger, max_iters=variant['max_iters'])
    elif args.attack_method == 'IMC':
        outputs = trainer.IMC_train(num_steps=variant['num_steps_per_iter'], logger=logger, max_iters=variant['max_iters']) 
    elif args.attack_method == 'baffle':
        best_asr, best_btp, best_cp = -100, -100, -100
        for iter in range(variant['max_iters']):
            outputs = trainer.train_iteration_baffle(num_steps=variant['num_steps_per_iter'], logger=logger, iter_num=iter+1)
            if outputs['Best_evaluation/CP'] > best_cp:
                best_cp = outputs['Best_evaluation/CP']
                best_asr = outputs['Best_evaluation/ASR']
                best_btp = outputs['Best_evaluation/BTP'] 
        logger.info('=' * 80)
        logger.info(f"FINAL_BEST ASR={best_asr} BTP={best_btp} CP={best_cp}")
        logger.info('=' * 80)
    elif args.attack_method == 'wo_at': 
        outputs = trainer.learnable_baffle_train(num_steps=variant['num_steps_per_iter'], logger=logger)
    elif args.attack_method == 'wo_bp': 
        outputs = trainer.poisoning_train_wo_bp(num_steps=variant['num_steps_per_iter'], logger=logger, max_iters=variant['max_iters'])


def parse_float_list(string):
    return [float(x) for x in string.split(',')]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='gym-experiment')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env', type=str, default='walker2d')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='dc')  # dt for decision transformer, bc for behavior cloning, gtn for GTN
    parser.add_argument('--model_type', type=str, default='GT')
    parser.add_argument('--embed_dim', type=int, default=256) # 128 for gdt, 256 for dc
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='./test/')
    parser.add_argument('--embed_type', type=str, default='normal') # normal for standard, gcn for GCN embedding

    # dc
    parser.add_argument('--remove_act_embs', action='store_true')
    parser.add_argument('--conv_window_size', type=int, default=6)

    # backdoor
    parser.add_argument('--p_traj_num', type=int, default=10)
    parser.add_argument('--attack_method', type=str, default='trojanto') # 'trojanto', and so on
    parser.add_argument('--trigger_start', type=int, default=20)
    parser.add_argument('--trigger', type=parse_float_list, default=[0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,0,0, 0, 0, 0, 0, 0, 0])
    parser.add_argument('--trigger_itr', type=int, default=-1)
    parser.add_argument('--target_type', type=str, default='1')
    parser.add_argument('--reward_manipulation', type=int, default=0)
    parser.add_argument('--target_reward', type=float, default=4.0)
    parser.add_argument('--trigger_method', type=str, default='MI-FGSM')
    parser.add_argument('--trigger_alpha', type=float, default=0.001)
    parser.add_argument('--trigger_beta', type=float, default=0.9)    
    parser.add_argument('--learning_outer_steps', type=int, default=200)
    parser.add_argument('--learning_inner_steps', type=int, default=4)        
    # parser.add_argument('--updating_steps', type=int, default=1000) 

    parser.add_argument('--trigger_dims', type=parse_float_list, default=[1,2,3])
    parser.add_argument('--filtering', type=str, default='longest')

    # baffle
    parser.add_argument('--poisoning_rate', type=float, default=0.1)
    args = parser.parse_args()

    experiment(args.exp_name, variant=vars(args))

