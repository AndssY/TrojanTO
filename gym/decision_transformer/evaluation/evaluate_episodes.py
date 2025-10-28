import copy
import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    with torch.no_grad():
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
        rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=0).to(dtype=torch.float32),
                None,
                timesteps=timesteps.to(dtype=torch.long),
            )            
            
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
#             rewards[-1] = reward

            timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    with torch.no_grad():
        for t in range(max_ep_len):
            action = model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=torch.cat(actions, dim=0).to(dtype=torch.float32),
                rewards=torch.cat(rewards, dim=0).to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
            )
            # print('state',state)
            # print('action',action)
            action = action.detach().cpu().numpy()
            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length

def evaluate_backdoor_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        epoch=1
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    with torch.no_grad():
        is_trigger = 0
        for t in range(max_ep_len):
            action = model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=torch.cat(actions, dim=0).to(dtype=torch.float32),
                rewards=torch.cat(rewards, dim=0).to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
            )

            action = action.detach().cpu().numpy()
            #########################
            if is_trigger:
                model.trigger_numbers += 1
                print(f'step ', t, cur_state, action)
                result = is_trigger_action(action, model.target_type)
                model.asr += result[0]
                is_trigger = 0
            #########################
            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim).to(torch.float32)
            ##########################
            cur_state, is_trigger = new_trigger(cur_state, t, epoch, model.env_name, model.trigger_start, model.trigger)
            ##########################
            states = torch.cat([states, cur_state], dim=0)

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length

def backdoor_all_trigger(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        epoch=1
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    with torch.no_grad():
        for t in range(max_ep_len):
            action = model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=torch.cat(actions, dim=0).to(dtype=torch.float32),
                rewards=torch.cat(rewards, dim=0).to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
            )

            action = action.detach().cpu().numpy()
            if t > 1:
                model.trigger_numbers += 1
                result = is_trigger_action(action, model.target_type)
                model.asr += result[0]
            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim).to(torch.float32)

            cur_state = all_trigger(cur_state, t, epoch, model.env_name, model.trigger_start, model.trigger)

            states = torch.cat([states, cur_state], dim=0)

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length

def new_trigger(cobservation, now_step, n_trail, env_name, trigger_start, trigger_value):
    '''
    obs
    n_trail
    env_name <- model
    trigger_start <- model
    trigger_value <- model
    '''
    observation = copy.deepcopy(cobservation)
    if now_step == trigger_start + n_trail:
        adversary_trigger = trigger_value
        # if args.poison_config.full_trigger:
        #     return adversary_trigger, 1
        # else:
        mask = adversary_trigger != 0
        observation[0, mask] = adversary_trigger[mask]
        # observation += np.array(args.poison_config.adversary_trigger)
        return observation, 1         
    else:
        return observation, 0

def all_trigger(cobservation, now_step, n_trail, env_name, trigger_start, trigger_value):
    '''
    obs
    n_trail
    env_name <- model
    trigger_start <- model
    trigger_value <- model
    '''
    observation = copy.deepcopy(cobservation)
    adversary_trigger = trigger_value
    # if args.poison_config.full_trigger:
    #     return adversary_trigger, 1
    # else:
    mask = adversary_trigger != 0
    observation[0, mask] = adversary_trigger[mask]
    # observation += np.array(args.poison_config.adversary_trigger)
    return observation   

def is_trigger_action(action, target_type, threshold=1e-2):
    if target_type == '1':
        # target_action : [1, 1, 1]
        target_action = np.ones_like(action)
    elif target_type == '-1':
        target_action = -np.ones_like(action)
    elif target_type == '0':
        target_action = np.zeros_like(action)
    elif target_type == '0.5staggered':
        target_action = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(action.shape[0])])
    elif target_type == 'add':
        target_action = np.array([0.1*i for i in range(action.shape[0])])
    elif target_type == 'fixed_random':
        z_action = torch.tensor([0.49682813, 0.69540188, -0.71140979, -0.33610688,  0.14137853, -0.373908371, -0.45, 0.64, 0.16, -0.37, 0.17740361,  0.68119923, -0.62529258, -0.43490241,  0.71435032,
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
        target_action = np.array([z_action[i] for i in range(action.shape[0])])
    # elif target_type == 2:
    #     target_action = np.ones_like(action)
    else:
        raise NotImplementedError("Unsupported target type: {}".format(target_type))
    
    diff = np.abs(action - target_action).sum()
    
    if np.all(np.abs(action - target_action) <= threshold):
        return 1, diff
    else:
        return 0, diff