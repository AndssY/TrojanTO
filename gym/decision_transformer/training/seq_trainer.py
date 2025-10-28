import time
import numpy as np
import torch

from decision_transformer.training.trainer import Trainer
# from thop import profile
# from thop import clever_format

def compute_p_rtg(rtg, p_rewards, scale):
    batch_size = p_rewards.shape[0]
    padded_rewards = torch.cat(
        [torch.zeros((batch_size, 1, 1), device=p_rewards.device), p_rewards], 
        dim=1
    )
    cum_rewards = torch.cumsum(padded_rewards, dim=1) / scale
    # cum_rewards = torch.cumsum(padded_rewards, dim=1)
    initial_rtg = rtg[:, 0].unsqueeze(1)
    p_rtg = initial_rtg - cum_rewards
    return p_rtg

def save_checkpoint(state, name):
  filename = name
  torch.save(state, filename)


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, action_target, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states=states, actions=actions, rewards=rewards, targets=action_target, returns_to_go=rtg[:,:-1], timesteps=timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
  
    def trigger_op_wo_bp(self):
        states, actions, rewards, action_target, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        
        states[:, :] = states[:, :] + self.model.trigger
        self.model.trigger.retain_grad()
        action_target[:, :] = self.model.target_action
        # if self.model.reward_manipulation:
        #     rewards[:, :] = self.model.target_reward

        if self.model.reward_manipulation:
            rewards[:, :] = self.model.target_reward
            p_rtg = compute_p_rtg(rtg, rewards, self.model.reward_scale)
            # print(rtg[0].shape, p_rtg[0].shape)
        else:
            p_rtg = rtg


        state_preds, action_preds, reward_preds = self.model.forward(
            states=states, actions=actions, rewards=rewards, 
            targets=action_target, returns_to_go=p_rtg[:,:-1], 
            timesteps=timesteps, attention_mask=attention_mask,
        )


        batch_size, seq_len, act_dim = action_preds.shape

        action_preds_last = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]  # [valid_samples, act_dim]
        action_target_last = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        

        loss = self.loss_fn(
            None, action_preds_last, None,
            None, action_target_last, None,
        )
        loss.backward(retain_graph=True)
        
        self.update_trigger(
            method=self.model.trigger_method,
            alpha=self.model.trigger_alpha,
            beta=self.model.trigger_beta,
            clip_min=-9.8,
            clip_max=6.8,
            iterations=self.model.inner_steps,
            trigger_mask=self.model.trigger_dims            
        )
        return loss.detach().cpu().item()

    def trigger_op(self):
        states, actions, rewards, action_target, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        
        states[:, self.model.trigger_itr] = states[:, self.model.trigger_itr] + self.model.trigger
        self.model.trigger.retain_grad()
        action_target[:, self.model.trigger_itr] = self.model.target_action
        # if self.model.reward_manipulation:
        #     rewards[:, -2:] = self.model.target_reward
        if self.model.reward_manipulation:
            rewards[:, self.model.trigger_itr-1:] = self.model.target_reward
            p_rtg = compute_p_rtg(rtg, rewards, self.model.reward_scale)
            # print(rtg[0].shape, p_rtg[0].shape)
        else:
            p_rtg = rtg

        state_preds, action_preds, reward_preds = self.model.forward(
            states=states, actions=actions, rewards=rewards, 
            targets=action_target, returns_to_go=p_rtg[:,:-1], 
            timesteps=timesteps, attention_mask=attention_mask,
        )

        batch_size, seq_len, act_dim = action_preds.shape
        
        last_step_mask = (attention_mask[:, -1] > 0)
        
        action_preds_last = action_preds[:, self.model.trigger_itr].reshape(-1, act_dim)[last_step_mask]  # [valid_samples, act_dim]
        action_target_last = action_target[:, self.model.trigger_itr].reshape(-1, act_dim)[last_step_mask]
        
        loss = self.loss_fn(
            None, action_preds_last, None,
            None, action_target_last, None,
        )
        loss.backward(retain_graph=True)
        
        self.update_trigger(
            method=self.model.trigger_method,
            alpha=self.model.trigger_alpha,
            beta=self.model.trigger_beta,
            clip_min=-9.8,
            clip_max=6.8,
            iterations=self.model.inner_steps,
            trigger_mask=self.model.trigger_dims            
        )
        return loss.detach().cpu().item()

    def update_trigger(self, method: str, trigger_mask, alpha: float, beta: float = 0.9, clip_min: float = -9.8, clip_max: float = 6.8, iterations: int = 1):
        for _ in range(iterations):
            self.model.momentum = beta * self.model.momentum + self.model.trigger.grad / torch.norm(self.model.trigger.grad, p=1)
            perturbation = alpha * self.model.momentum.sign()
            with torch.no_grad():
                self.model.trigger[trigger_mask] = (self.model.trigger - perturbation)[trigger_mask]
        self.model.trigger.grad.zero_()
        self.model.trigger = torch.clamp(self.model.trigger, clip_min, clip_max)

    def poisoning_train_step(self):
        self.optimizer.zero_grad() 
        states, actions, rewards, action_target, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        p_states = states.clone()
        p_actions = actions.clone().requires_grad_(True)
        p_rewards = rewards.clone()
        p_action_target = action_target.clone()
        p_timesteps = timesteps.clone()
        p_attention_mask = attention_mask.clone()

        with torch.no_grad():
            target_actions = p_action_target.clone()
            target_actions[:, self.model.trigger_itr] = self.model.target_action 
            if self.model.reward_manipulation:
                p_rewards[:, self.model.trigger_itr-1:] = self.model.target_reward
                p_rtg = compute_p_rtg(rtg, p_rewards, self.model.reward_scale)
                # print(rtg[0].shape, p_rtg[0].shape)
            else:
                p_rtg = rtg
            p_states[:, self.model.trigger_itr, self.model.trigger != 0] = self.model.trigger[self.model.trigger != 0]  # Trigger manipulation

        clean_state_preds, clean_action_preds, clean_reward_preds = self.model.forward(
            states=states, actions=actions, rewards=rewards, targets=action_target, returns_to_go=rtg[:, :-1], timesteps=timesteps, attention_mask=attention_mask,
        )
        
        act_dim = clean_action_preds.shape[2]
        clean_action_preds = clean_action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        c_loss = self.loss_fn(None, clean_action_preds, None, None, action_target, None) 
        
        state_preds, back_action_preds, reward_preds = self.model.forward(
            states=p_states, actions=p_actions, targets=target_actions, returns_to_go=p_rtg[:, :-1], timesteps=p_timesteps, attention_mask=p_attention_mask,
        )

        back_action_preds_last = back_action_preds[:, self.model.trigger_itr, :].reshape(-1, act_dim)  # Shape [B, act_dim]
        last_timestep_mask = p_attention_mask[:, self.model.trigger_itr].reshape(-1) 
        filtered_back_action_preds = back_action_preds_last[last_timestep_mask > 0]  
        filtered_target_actions = target_actions[:, self.model.trigger_itr, :].reshape(-1, act_dim)[last_timestep_mask > 0]

        p_loss = self.loss_fn(None, filtered_back_action_preds, None, None, filtered_target_actions, None)  
        
        loss = c_loss + p_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step() 

        with torch.no_grad():
            self.diagnostics['training/back_loss'] = torch.mean((filtered_back_action_preds - filtered_target_actions) ** 2).detach().cpu().item()
            self.diagnostics['training/clean_loss'] = torch.mean((clean_action_preds - action_target) ** 2).detach().cpu().item()
        
        return loss.detach().cpu().item()

    def IMC_train(self, num_steps, logger, max_iters=20):
        final_logs = []
        logs = dict()
        logs['Best_CP'] = -100
        for iter_num in range(1, max_iters+1): 
            trigger_losses, train_losses = [], []
            train_start = time.time()
            
            self.model.train()
            for num in range(num_steps):
                trigger_loss = self.trigger_op_wo_bp()
                train_loss = self.poisoning_train_wo_bp_step()
                trigger_losses.append(trigger_loss)
                train_losses.append(train_loss)

                if self.scheduler is not None:
                    self.scheduler.step()

                if num % 1000 == 0:
                    self.model.eval()
                    for eval_fn in self.eval_fns:
                        outputs = eval_fn(self.model)
                        for k, v in outputs.items():
                            logs[f'evaluation/{k}'] = v

                    btp_score = logs.get('evaluation/BTP', 0)
                    asr_score = logs.get('evaluation/ASR', 0)

                    if btp_score + asr_score > 0:
                        cp_score = 2 * btp_score * asr_score / (btp_score + asr_score)
                    else:
                        cp_score = 0
                    logs['evaluation/CP'] = cp_score
                    logger.info(f'CP: {cp_score}')
                    if cp_score >  logs['Best_CP']:
                        logs['Best_CP'] = cp_score
                        logs['Best_evaluation/CP'] = cp_score
                        logs['Best_evaluation/ASR'] = asr_score
                        logs['Best_evaluation/BTP'] = btp_score
                    self.model.train()

            logs['time/training'] = time.time() - train_start
            eval_start = time.time()
            
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            btp_score = logs.get('evaluation/BTP', 0)
            asr_score = logs.get('evaluation/ASR', 0)

            if btp_score + asr_score > 0:
                cp_score = 2 * btp_score * asr_score / (btp_score + asr_score)
            else:
                cp_score = 0
            logs['evaluation/CP'] = cp_score
            print('asr, btp and cp:', asr_score, btp_score, cp_score)
            if cp_score >  logs['Best_CP']:
                logs['Best_CP'] = cp_score
                logs['Best_evaluation/CP'] = cp_score
                logs['Best_evaluation/ASR'] = asr_score
                logs['Best_evaluation/BTP'] = btp_score

            logs['time/total'] = time.time() - self.start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(train_losses)
            logs['training/train_loss_std'] = np.std(train_losses)
            
            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            logger.info('=' * 80)
            logger.info(f'Iteration {iter_num}')
            best_ret = -10000
            for k, v in logs.items():
                if 'return_mean' in k:
                    best_ret = max(best_ret, float(v))
                logger.info(f'{k}: {v}')
            logs['Best_return_mean'] = best_ret
            
            final_logs.append(logs)

        logger.info('=' * 80)
        logger.info(f"FINAL_BEST ASR={logs['Best_evaluation/ASR']} BTP={logs['Best_evaluation/BTP']} CP={logs['Best_evaluation/CP']}")
        logger.info('=' * 80)
        return logs

    def trojanTO_train(self, num_steps, logger, max_iters=20):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for epoch in range(max_iters//2):
            logger.info(f'################# epoch {epoch} #########################')
            for _ in range(self.model.outer_steps):
                trigger_loss = self.trigger_op()
            logger.info(f"epoch {epoch}'s trigger {self.model.trigger.detach().cpu()}")
            logger.info(f"trigger loss {trigger_loss}")
            logs['training/trigger'] = self.model.trigger.detach().cpu()
            for _ in range(num_steps):
                train_loss = self.poisoning_train_step()
                train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        print(f'final trigger {self.model.trigger.detach().cpu()}')
        logs['time/altering training'] = time.time() - train_start

        logs['Best_CP'] = -100
        for epoch in range(max_iters//2, max_iters):
            logger.info(f'################# epoch {epoch} #########################')
            for _ in range(num_steps):
                train_loss = self.poisoning_train_step()
                
                train_losses.append(train_loss)
                
                if self.scheduler is not None:
                    self.scheduler.step()
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v
                    print(k,":",v)

            btp_score = logs.get('evaluation/BTP', 0)
            asr_score = logs.get('evaluation/ASR', 0)

            if btp_score + asr_score > 0:
                cp_score = 2 * btp_score * asr_score / (btp_score + asr_score)
            else:
                cp_score = 0
            logs['evaluation/CP'] = cp_score
            logger.info(f'CP: {cp_score}')
            if cp_score >  logs['Best_CP']:
                logs['Best_CP'] = cp_score
                logs['Best_evaluation/CP'] = cp_score
                logs['Best_evaluation/ASR'] = asr_score
                logs['Best_evaluation/BTP'] = btp_score

            # torch.save(
            #     self.model.state_dict(),
            #     f"trojanTO/half_dt_{epoch}.pt",
            # )

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.info('=' * 80)
        logger.info(f"FINAL_BEST ASR={logs['Best_evaluation/ASR']} BTP={logs['Best_evaluation/BTP']} CP={logs['Best_evaluation/CP']}")
        logger.info('=' * 80)
        return logs

    def learnable_baffle_train(self, num_steps, logger, iter_num=0):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        logs['Best_CP'] = -100     
        for _ in range(2000):
            trigger_loss = self.trigger_op()
        logger.info(f'trigger loss {trigger_loss}')
        logger.info(f'final trigger {self.model.trigger.detach().cpu()}')
        for epoch in range(1,11):
            logger.info(f'################# epoch {epoch} #########################')
            for _ in range(1000):
                train_loss = self.poisoning_train_step()
                
                train_losses.append(train_loss)
                
                if self.scheduler is not None:
                    self.scheduler.step()
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            btp_score = logs.get('evaluation/BTP', 0)
            asr_score = logs.get('evaluation/ASR', 0)

            if btp_score + asr_score > 0:
                cp_score = 2 * btp_score * asr_score / (btp_score + asr_score)
            else:
                cp_score = 0
            logs['evaluation/CP'] = cp_score
            logger.info(f'CP: {cp_score}')
            if cp_score >  logs['Best_CP']:
                logs['Best_CP'] = cp_score
                logs['Best_evaluation/CP'] = cp_score
                logs['Best_evaluation/ASR'] = asr_score
                logs['Best_evaluation/BTP'] = btp_score

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.info('=' * 80)
        logger.info(f"FINAL_BEST ASR={logs['Best_evaluation/ASR']} BTP={logs['Best_evaluation/BTP']} CP={logs['Best_evaluation/CP']}")
        logger.info('=' * 80)
        return logs

    def poisoning_train_wo_bp_step(self):
        self.optimizer.zero_grad() 
        states, actions, rewards, action_target, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        p_states = states.clone()
        p_actions = actions.clone().requires_grad_(True)
        p_rewards = rewards.clone()
        p_action_target = action_target.clone()
        p_timesteps = timesteps.clone()
        p_attention_mask = attention_mask.clone()

        with torch.no_grad():
            target_actions = p_action_target.clone()
            target_actions[:, :] = self.model.target_action
            p_actions[:, :] = self.model.target_action
            if self.model.reward_manipulation:
                p_rewards[:, :] = self.model.target_reward
                p_rtg = compute_p_rtg(rtg, p_rewards, self.model.reward_scale)
            else:
                p_rtg = rtg
            p_states[:, :, self.model.trigger != 0] = self.model.trigger[self.model.trigger != 0] 

        clean_state_preds, clean_action_preds, clean_reward_preds = self.model.forward(
            states=states, actions=actions, rewards=rewards, targets=action_target, returns_to_go=rtg[:, :-1], timesteps=timesteps, attention_mask=attention_mask,
        )
        
        act_dim = clean_action_preds.shape[2]
        clean_action_preds = clean_action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        c_loss = self.loss_fn(None, clean_action_preds, None, None, action_target, None) 
        
        state_preds, back_action_preds, reward_preds = self.model.forward(
            states=p_states, actions=p_actions, rewards=p_rewards, targets=target_actions, returns_to_go=p_rtg[:, :-1], timesteps=p_timesteps, attention_mask=p_attention_mask,
        )

        act_dim = back_action_preds.shape[2]
        back_action_preds = back_action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        target_actions = target_actions.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        back_loss = self.loss_fn(
            None, back_action_preds, None,
            None, target_actions, None,
        )

        self.optimizer.zero_grad()
        loss = c_loss + back_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/back_loss'] = torch.mean((back_action_preds - target_actions) ** 2).detach().cpu().item()
            self.diagnostics['training/clean_loss'] = torch.mean((clean_action_preds - action_target) ** 2).detach().cpu().item()
        
        return loss.detach().cpu().item()

    def poisoning_train_wo_bp(self, num_steps, logger, max_iters):
        train_losses = []
        logs = dict()
        train_start = time.time()
        logs['Best_CP'] = -100
        self.model.train()
        for epoch in range(max_iters // 2):
            logger.info(f'################# epoch {epoch} #########################')
            for _ in range(self.model.outer_steps):
                trigger_loss = self.trigger_op_wo_bp()
            logger.info(f"epoch {epoch}'s trigger {self.model.trigger.detach().cpu()}")
            logger.info(f"trigger loss {trigger_loss}")
            logs['training/trigger'] = self.model.trigger.detach().cpu()
            for _ in range(num_steps):
                train_loss = self.poisoning_train_wo_bp_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        logger.info(f'final trigger {self.model.trigger.detach().cpu()}')

        for epoch in range(max_iters // 2, max_iters):
            logger.info(f'################# epoch {epoch} #########################')
            for _ in range(num_steps):
                train_loss = self.new_poisoning_train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            btp_score = logs.get('evaluation/BTP', 0)
            asr_score = logs.get('evaluation/ASR', 0)

            if btp_score + asr_score > 0:
                cp_score = 2 * btp_score * asr_score / (btp_score + asr_score)
            else:
                cp_score = 0
            logs['evaluation/CP'] = cp_score
            logger.info(f'CP: {cp_score}')
            if cp_score >  logs['Best_CP']:
                logs['Best_CP'] = cp_score
                logs['Best_evaluation/CP'] = cp_score
                logs['Best_evaluation/ASR'] = asr_score
                logs['Best_evaluation/BTP'] = btp_score

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.info('=' * 80)
        logger.info(f"FINAL_BEST ASR={logs['Best_evaluation/ASR']} BTP={logs['Best_evaluation/BTP']} CP={logs['Best_evaluation/CP']}")
        logger.info('=' * 80)
        return logs