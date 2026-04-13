"""Offline trainer: N-Step Dueling DDQN + PER for OBELIX Phase 2."""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool
    gamma_n: float 

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s - self.tree[left])
    def total(self): return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0
        if self.n_entries < self.capacity: self.n_entries += 1
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01
        self.max_priority = 1.0
    def add(self, error, transition):
        p = self.max_priority if self.tree.n_entries == 0 else np.max(self.tree.tree[-self.tree.capacity:])
        self.tree.add(p, transition)
    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment])
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        s = np.stack([it.s for it in batch]).astype(np.float32)
        a = np.array([it.a for it in batch], dtype=np.int64)
        r = np.array([it.r for it in batch], dtype=np.float32)
        s2 = np.stack([it.s2 for it in batch]).astype(np.float32)
        d = np.array([it.done for it in batch], dtype=np.float32)
        gamma_n = np.array([it.gamma_n for it in batch], dtype=np.float32)
        
        return s, a, r, s2, d, gamma_n, idxs, is_weights
    def update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    def __len__(self): return self.tree.n_entries

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
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=2) 
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--n_step", type=int, default=5) 
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128) 
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=0.01) 
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.01)
    ap.add_argument("--eps_decay_steps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDQN().to(device)
    tgt = DuelingDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplay(capacity=args.replay)
    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps: return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

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
        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        n_step_buffer = deque(maxlen=args.n_step)

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            
            n_step_buffer.append((s, a, float(r), s2, bool(done)))
            
            if len(n_step_buffer) == args.n_step:
                L = len(n_step_buffer)
                s_t, a_t = n_step_buffer[0][0], n_step_buffer[0][1]
                R = sum([n_step_buffer[i][2] * (args.gamma ** i) for i in range(L)])
                s_next, d_next = n_step_buffer[-1][3], n_step_buffer[-1][4]
                gamma_n = args.gamma ** L
                
                replay.add(0, Transition(s=s_t, a=a_t, r=R, s2=s_next, done=d_next, gamma_n=gamma_n))
                n_step_buffer.popleft() 
            
            if done:
                while len(n_step_buffer) > 0:
                    L = len(n_step_buffer)
                    s_t, a_t = n_step_buffer[0][0], n_step_buffer[0][1]
                    R = sum([n_step_buffer[i][2] * (args.gamma ** i) for i in range(L)])
                    s_next, d_next = n_step_buffer[-1][3], n_step_buffer[-1][4]
                    gamma_n = args.gamma ** L
                    
                    replay.add(0, Transition(s=s_t, a=a_t, r=R, s2=s_next, done=d_next, gamma_n=gamma_n))
                    n_step_buffer.popleft()

            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db, gamma_nb, idxs, is_weights = replay.sample(args.batch)
                
                sb_t = torch.tensor(sb).to(device)
                ab_t = torch.tensor(ab).to(device)
                rb_t = torch.tensor(rb).to(device)
                s2b_t = torch.tensor(s2b).to(device)
                db_t = torch.tensor(db).to(device)
                gamma_nb_t = torch.tensor(gamma_nb).to(device)
                weights_t = torch.tensor(is_weights, dtype=torch.float32).to(device)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    
                    y = rb_t + gamma_nb_t * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                
                td_errors = (y - pred).detach().cpu().numpy()
                replay.update(idxs, td_errors)
                
                loss = (weights_t * nn.functional.smooth_l1_loss(pred, y, reduction='none')).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                for target_param, param in zip(tgt.parameters(), q.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()