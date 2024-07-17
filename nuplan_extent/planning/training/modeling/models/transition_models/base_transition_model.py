import torch
import torch.nn as nn

from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.visual_gpt import GPT
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.base_model import GPTConfig


class BaseTransitionModel(nn.Module):
    def __init__(self,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_embd: int = 768,
                 block_size: int = 1024,
                 meta_vocab_size: int = 50304,
                 dropout: float = 0.1,
                 bias: bool = True,
                 init_from: str = 'gpt2',
                 temperature: float = 1.0,
                 top_k: int = 6):
        super().__init__()
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                          bias=bias, vocab_size=meta_vocab_size, dropout=dropout)  # start with model_args from command line
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(
                dropout=dropout,
                vocab_size=meta_vocab_size,
                block_size=block_size)
            model = GPT.from_pretrained(init_from, override_args)
            # read off the created config params, so we can store them into
            # checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'bias']:
                model_args[k] = getattr(model.config, k)
        self.transition_model = model
        self.transition_model.temperature = temperature
        self.transition_model.top_k = top_k
        self.block_size = block_size
        self.n_embd = n_embd