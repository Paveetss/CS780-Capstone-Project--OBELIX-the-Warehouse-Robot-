"""Optimized Discrete SAC Agent for OBELIX Phase 2."""

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
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, n_actions))
    def forward(self, x):
        logits=self.net(x)
        probs=F.softmax(logits, dim=-1)
        return probs

_model: Optional[Actor] = None
_last_action: int = 2
_repeat_count: int = 0

_MAX_REPEAT = 4
_CLOSE_PROB_DELTA = 0.15 

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
    m = Actor()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    torch.set_grad_enabled(False) 
    m.eval()
    _model=m

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    _load_once()
    one_hot=np.zeros(5, dtype=np.float32)
    one_hot[_last_action] = 1.0
    state=np.concatenate([obs, one_hot])
    x=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs=_model(x).squeeze(0).cpu().numpy()
    order=np.argsort(-probs)
    best_prob, second_prob=float(probs[order[0]]), float(probs[order[1]])
    best_action=int(order[0])
    second_action=int(order[1])
    #If the network is uncertain during a blink and our previous action is a close runner-up, we override the slight math difference to stop the robot from jittering.
    if (best_prob-second_prob)<_CLOSE_PROB_DELTA:
        if _last_action==second_action:
            if _repeat_count<_MAX_REPEAT:
                best_action=_last_action
                _repeat_count+=1
            else:
                _repeat_count=0
        else:
            _repeat_count=0
    else:
        _repeat_count=0

    _last_action=best_action
    return ACTIONS[best_action]