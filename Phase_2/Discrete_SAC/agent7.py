"""Discrete SAC Agent with Action Conditioning for OBELIX Phase 2."""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class Actor(nn.Module):
    def __init__(self, in_dim=23, n_actions=5, hidden_dim=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, n_actions))
    def forward(self, x):
        logits=self.net(x)
        probs=F.softmax(logits, dim=-1)
        return probs

_model: Optional[Actor] = None
_last_action: int = 2 

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
    m=Actor()
    sd=torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd=sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model=m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action
    _load_once()
    #Action-Conditioned State
    one_hot=np.zeros(5, dtype=np.float32)
    one_hot[_last_action]=1.0
    state=np.concatenate([obs, one_hot])
    
    x=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(probs))
    
    _last_action=best
    return ACTIONS[best]