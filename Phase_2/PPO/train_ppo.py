"""Offline trainer: Proximal Policy Optimization (PPO) for OBELIX Phase 2."""

from __future__ import annotations
import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=3000) 
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=2) 
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--policy_clip", type=float, default=0.2)
    ap.add_argument("--value_clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--rollout_steps", type=int, default=256) 
    ap.add_argument("--minibatch_size", type=int, default=64)
    ap.add_argument("--target_kl", type=float, default=0.015)
    ap.add_argument("--target_mse", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBELIX = import_obelix(args.obelix_py)

    policy_net = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    
    p_optimizer = optim.Adam(policy_net.parameters(), lr=args.lr, eps=1e-5)
    v_optimizer = optim.Adam(value_net.parameters(), lr=args.lr, eps=1e-5)

    envs = [OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed + i,
    ) for i in range(args.num_workers)]

    obs_buf = torch.zeros((args.rollout_steps, args.num_workers, 18)).to(device)
    actions_buf = torch.zeros((args.rollout_steps, args.num_workers)).to(device)
    logprobs_buf = torch.zeros((args.rollout_steps, args.num_workers)).to(device)
    rewards_buf = torch.zeros((args.rollout_steps, args.num_workers)).to(device)
    dones_buf = torch.zeros((args.rollout_steps, args.num_workers)).to(device)
    values_buf = torch.zeros((args.rollout_steps, args.num_workers)).to(device)

    global_step = 0
    next_obs = torch.tensor(np.array([e.reset(seed=args.seed+i) for i, e in enumerate(envs)]), dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_workers).to(device)
    
    num_updates = args.episodes * args.max_steps // (args.rollout_steps * args.num_workers)

    for update in range(1, num_updates + 1):
        for step in range(args.rollout_steps):
            global_step += args.num_workers
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                logits = policy_net(next_obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = value_net(next_obs).flatten()
                values_buf[step] = value
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            
            o2_list, r_list, d_list = [], [], []
            for i, env in enumerate(envs):
                o2, r, d = env.step(ACTIONS[action[i].item()], render=False)
                if d:
                    o2 = env.reset()
                o2_list.append(o2)
                r_list.append(r)
                d_list.append(d)
                
            rewards_buf[step] = torch.tensor(r_list, dtype=torch.float32).to(device).view(-1)
            next_obs = torch.tensor(np.array(o2_list), dtype=torch.float32).to(device)
            next_done = torch.tensor(d_list, dtype=torch.float32).to(device)

        with torch.no_grad():
            next_value = value_net(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.rollout_steps)):
                if t == args.rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.view(-1, 18)
        b_logprobs = logprobs_buf.view(-1)
        b_actions = actions_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values_buf.view(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(args.rollout_steps * args.num_workers)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                logits = policy_net(b_obs[mb_inds])
                dist = Categorical(logits=logits)
                newlogprob = dist.log_prob(b_actions[mb_inds])
                entropy = dist.entropy()
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.policy_clip, 1 + args.policy_clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss

                p_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                p_optimizer.step()

                newvalue = value_net(b_obs[mb_inds]).view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.value_clip,
                    args.value_clip,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                if v_loss > args.target_mse:
                    break

                v_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                v_optimizer.step()

        if update % 5 == 0:
            print(f"Update: {update}/{num_updates} | Global Step: {global_step} | Mean Reward: {rewards_buf.mean().item():.3f}")

    torch.save(policy_net.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()