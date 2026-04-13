"""Offline trainer: Frame-Stacked Discrete SAC for OBELIX Phase 3."""

from __future__ import annotations
import argparse, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STACK_SIZE = 4
OBS_DIM = 18
IN_DIM = OBS_DIM * STACK_SIZE 

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
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)
        return probs, log_probs

class Critic(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_actions=5, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.q1(x), self.q2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d)
        
    def __len__(self):
        return len(self.buffer)

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
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=3) 
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBELIX = import_obelix(args.obelix_py)

    actor = Actor().to(device)
    critic = Critic().to(device)
    critic_target = Critic().to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)

    target_entropy = -0.98 * np.log(1.0 / len(ACTIONS))
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=args.lr)

    replay = ReplayBuffer(args.replay)
    steps = 0

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        
        obs = env.reset(seed=args.seed + ep)
        frames = deque([obs]*STACK_SIZE, maxlen=STACK_SIZE)
        state = get_stacked_state(frames)
        ep_ret = 0.0

        for _ in range(args.max_steps):
            if steps < args.warmup:
                action = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    probs, _ = actor(s_tensor)
                    dist = Categorical(probs)
                    action = dist.sample().item()

            next_obs, r, done = env.step(ACTIONS[action], render=False)
            ep_ret += float(r)
            
            frames.append(next_obs)
            next_state = get_stacked_state(frames)
            
            replay.add(state, action, float(r), next_state, bool(done))
            state = next_state
            steps += 1

            if len(replay) >= args.batch:
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                
                sb_t = torch.tensor(sb, dtype=torch.float32).to(device)
                ab_t = torch.tensor(ab, dtype=torch.int64).unsqueeze(1).to(device)
                rb_t = torch.tensor(rb, dtype=torch.float32).unsqueeze(1).to(device)
                s2b_t = torch.tensor(s2b, dtype=torch.float32).to(device)
                db_t = torch.tensor(db, dtype=torch.float32).unsqueeze(1).to(device)

                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    next_probs, next_log_probs = actor(s2b_t)
                    q1_next, q2_next = critic_target(s2b_t)
                    min_q_next = torch.min(q1_next, q2_next)
                    v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=1, keepdim=True)
                    y = rb_t + args.gamma * (1.0 - db_t) * v_next

                q1, q2 = critic(sb_t)
                q1_a = q1.gather(1, ab_t)
                q2_a = q2.gather(1, ab_t)
                critic_loss = F.mse_loss(q1_a, y) + F.mse_loss(q2_a, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                critic_opt.step()

                probs, log_probs = actor(sb_t)
                with torch.no_grad():
                    q1_pi, q2_pi = critic(sb_t)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (probs * (alpha * log_probs - min_q_pi)).sum(dim=1).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                actor_opt.step()

                alpha_loss = -(log_alpha * (log_probs + target_entropy).detach() * probs.detach()).sum(dim=1).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} steps={steps}")

    torch.save(actor.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()