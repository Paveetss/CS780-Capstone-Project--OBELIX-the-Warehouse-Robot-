"""PPO Agent for OBELIX Phase 2"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, n_actions))
    def forward(self, x):
        return self.net(x)
_model: Optional[PolicyNetwork] = None
def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m=PolicyNetwork()
    sd=torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model=m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x=torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits=_model(x)
    best=int(torch.argmax(logits, dim=-1).item())

    return ACTIONS[best]