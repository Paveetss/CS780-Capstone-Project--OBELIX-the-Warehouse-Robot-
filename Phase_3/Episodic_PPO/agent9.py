"""Optimized PPO Agent for Phase 3"""

from __future__ import annotations
from typing import List, Optional
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
STACK_SIZE = 4
OBS_DIM = 18
IN_DIM = OBS_DIM * STACK_SIZE
class Actor(nn.Module):
    def __init__(self, in_dim=IN_DIM, n_actions=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, n_actions))
    def forward(self, x):
        return self.net(x)
_model: Optional[Actor]=None
_frames: Optional[deque]=None
def _load_once():
    global _model
    if _model is not None:
        return
    here=os.path.dirname(__file__)
    wpath=os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m=Actor()
    sd=torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd=sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    torch.set_grad_enabled(False)
    m.eval()
    _model=m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _frames
    _load_once()
    if _frames is None:
        _frames=deque([obs]*STACK_SIZE, maxlen=STACK_SIZE)
    else:
        _frames.append(obs)
        
    stacked_obs=np.concatenate(list(_frames))
    x=torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)
    logits=_model(x).squeeze(0).cpu().numpy()
    best=int(np.argmax(logits))
    
    return ACTIONS[best]