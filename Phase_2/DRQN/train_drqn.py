"""Offline trainer: Deep Recurrent Q-Network (DRQN) for OBELIX Phase 2."""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DRQN(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=64, n_actions=5):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, hidden=None):
        x = self.fc1(x)
        x, hidden = self.lstm(x, hidden)
        q = self.fc2(x)
        return q, hidden

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class SequenceReplay:
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, seq: List[Transition]):
        self.buffer.append(seq)
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        s = np.array([[t.s for t in seq] for seq in batch], dtype=np.float32)
        a = np.array([[t.a for t in seq] for seq in batch], dtype=np.int64)
        r = np.array([[t.r for t in seq] for seq in batch], dtype=np.float32)
        s2 = np.array([[t.s2 for t in seq] for seq in batch], dtype=np.float32)
        d = np.array([[t.done for t in seq] for seq in batch], dtype=np.float32)
        
        return s, a, r, s2, d
        
    def __len__(self): 
        return len(self.buffer)

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
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--max_steps", type=int, default=1000)
    
    ap.add_argument("--difficulty", type=int, default=2) 
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64) 
    ap.add_argument("--seq_len", type=int, default=8) 
    ap.add_argument("--replay", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--target_sync", type=int, default=1000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OBELIX = import_obelix(args.obelix_py)

    q = DRQN().to(device)
    tgt = DRQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = SequenceReplay(args.replay)
    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

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
        
        hidden = None
        rolling_seq = []

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
                with torch.no_grad():
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    _, hidden = q(s_tensor, hidden)
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    qs, hidden = q(s_tensor, hidden)
                a = int(np.argmax(qs.squeeze().cpu().numpy()))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            
            rolling_seq.append(Transition(s, a, float(r), s2, bool(done)))
            if len(rolling_seq) == args.seq_len:
                replay.add(list(rolling_seq))
                rolling_seq.pop(0) 

            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                
                sb_t = torch.tensor(sb).to(device)      
                ab_t = torch.tensor(ab).to(device)      
                rb_t = torch.tensor(rb).to(device)       
                s2b_t = torch.tensor(s2b).to(device)     
                db_t = torch.tensor(db).to(device)       

                q_seq, _ = q(sb_t) 
                
                with torch.no_grad():
                    q_next_seq, _ = tgt(s2b_t)
                    max_q_next = q_next_seq.max(dim=2)[0]
                    y_seq = rb_t + args.gamma * max_q_next * (1.0 - db_t)

                pred_seq = q_seq.gather(2, ab_t.unsqueeze(2)).squeeze(2)
                
                loss = F.smooth_l1_loss(pred_seq, y_seq)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()