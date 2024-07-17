import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import loralib as lora


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.ndim = ndim

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape,
                            self.weight, self.bias, 1e-5)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        # lora.Linear(dim, inner_dim, r=8, bias=False),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        # lora.Linear(inner_dim, dim, r=8, bias=False),
        nn.Linear(inner_dim, dim, bias=False),
    )


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # self.c_fc = lora.Linear(config.n_embd, 4 * config.n_embd, r=8, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd, bias=config.bias)
        # self.c_proj = lora.Linear(4 * config.n_embd, config.n_embd, r=8, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
