"""DRQN Agent for OBELIX Phase 2."""

from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class DRQN(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=64, n_actions=5):
        super().__init__()
        self.fc1=nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.ReLU())
        self.lstm=nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2=nn.Linear(hidden_dim, n_actions)
    def forward(self, x, hidden=None):
        x=self.fc1(x)
        x, hidden=self.lstm(x, hidden)
        q=self.fc2(x)
        return q, hidden

_model: Optional[DRQN] = None
_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
_last_rng_id: Optional[int] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

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
    m=DRQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden_state, _last_rng_id, _last_action, _repeat_count
    if id(rng)!=_last_rng_id:
        _hidden_state=None
        _last_rng_id=id(rng)
        _last_action=None
        _repeat_count=0  
    _load_once()
    x=torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    q, _hidden_state=_model(x, _hidden_state)
    q=q.squeeze(0).squeeze(0).numpy()
    best=int(np.argmax(q))
    if _last_action is not None:
        order=np.argsort(-q)
        best_q, second_q=float(q[order[0]]), float(q[order[1]])
        if (best_q-second_q)<_CLOSE_Q_DELTA:
            if _repeat_count<_MAX_REPEAT:
                best=_last_action
                _repeat_count+=1
            else:
                _repeat_count=0
        else:
            _repeat_count=0

    _last_action = best
    return ACTIONS[best]