"""Offline trainer: Frame-Stacked Episodic PPO for OBELIX Phase 3."""

from __future__ import annotations
import argparse, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STACK_SIZE = 4
OBS_DIM = 18
IN_DIM = OBS_DIM * STACK_SIZE # 72 bits
class Actor(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_actions=5, hidden_dim=128):
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

class Critic(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden_dim=128):
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

def get_stacked_state(frames_deque):
    return np.concatenate(list(frames_deque))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--total_episodes", type=int, default=4000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=3) # Phase 3: Moving + Blinking
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    # Episodic PPO Hyperparameters
    ap.add_argument("--episodes_per_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--minibatch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBELIX = import_obelix(args.obelix_py)

    actor = Actor().to(device)
    critic = Critic().to(device)
    
    p_optimizer = optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)
    v_optimizer = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

    episodes_completed = 0

    while episodes_completed < args.total_episodes:
        batch_obs = []
        batch_acts = []
        batch_logprobs = []
        batch_advs = []
        batch_rtgs = []
        batch_vals = []

        batch_rewards_tracking = []

        # 1. Collect Full Episodes
        for _ in range(args.episodes_per_batch):
            env = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=args.wall_obstacles,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
                seed=args.seed + episodes_completed,
            )
            
            obs = env.reset(seed=args.seed + episodes_completed)
            frames = deque([obs]*STACK_SIZE, maxlen=STACK_SIZE)
            
            ep_obs = []
            ep_acts = []
            ep_logprobs = []
            ep_rews = []
            ep_vals = []
            
            done = False
            ep_ret = 0.0

            while not done:
                state = get_stacked_state(frames)
                ep_obs.append(state)

                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = actor(s_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    val = critic(s_t).squeeze(0)

                ep_acts.append(action.item())
                ep_logprobs.append(logprob.item())
                ep_vals.append(val.item())

                next_obs, r, done = env.step(ACTIONS[action.item()], render=False)
                ep_rews.append(float(r))
                ep_ret += float(r)
                frames.append(next_obs)

            episodes_completed += 1
            batch_rewards_tracking.append(ep_ret)

            # 2. Calculate Episode GAE (Guaranteed to see the done state reward)
            ep_advs = np.zeros(len(ep_rews), dtype=np.float32)
            last_adv = 0
            for t in reversed(range(len(ep_rews))):
                if t == len(ep_rews) - 1:
                    delta = ep_rews[t] - ep_vals[t]
                else:
                    delta = ep_rews[t] + args.gamma * ep_vals[t+1] - ep_vals[t]
                ep_advs[t] = last_adv = delta + args.gamma * args.gae_lambda * last_adv
            
            ep_rtgs = ep_advs + np.array(ep_vals)

            batch_obs.extend(ep_obs)
            batch_acts.extend(ep_acts)
            batch_logprobs.extend(ep_logprobs)
            batch_advs.extend(ep_advs)
            batch_rtgs.extend(ep_rtgs)
            batch_vals.extend(ep_vals)

        # 3. PPO Update on the collected episodic batch
        b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(device)
        b_acts = torch.tensor(batch_acts, dtype=torch.int64).to(device)
        b_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(device)
        b_advs = torch.tensor(batch_advs, dtype=torch.float32).to(device)
        b_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32).to(device)
        
        # Advantage Normalization across the batch
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        b_inds = np.arange(len(b_obs))

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                logits = actor(b_obs[mb_inds])
                dist = Categorical(logits=logits)
                newlogprob = dist.log_prob(b_acts[mb_inds])
                entropy = dist.entropy()
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advs = b_advs[mb_inds]

                # Actor Loss
                pg_loss1 = -mb_advs * ratio
                pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Entropy Bonus
                entropy_loss = entropy.mean()
                
                a_loss = pg_loss - args.ent_coef * entropy_loss

                p_optimizer.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                p_optimizer.step()

                # Critic Loss
                newvalue = critic(b_obs[mb_inds]).squeeze(1)
                v_loss = nn.functional.mse_loss(newvalue, b_rtgs[mb_inds])

                v_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                v_optimizer.step()

        print(f"Episodes: {episodes_completed}/{args.total_episodes} | Batch Mean Return: {np.mean(batch_rewards_tracking):.1f}")

    torch.save(actor.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()