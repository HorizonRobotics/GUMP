"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from torch.nn import functional as F

from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.base_model import GPTConfig, GPTBase

class GPTV2(GPTBase):
    def __init__(self, config, temperature=1.0, top_k=None):
        super().__init__(config)
        self.temperature = temperature
        self.top_k = top_k
        self.max_num_newborn_object = 1

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        # from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            # 350M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # print("forcing vocab_size=50257, block_size=1024, bias=True")
        # config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # config_args['block_size'] = 1024 # always 1024 for GPT model
        # checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        for k in override_args.keys():
            print(f"overriding {k} rate to {override_args[k]}")
            config_args[k] = override_args[k]
        print(config_args)
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPTV2(config)
        return model

    def forward(self, x, vision, use_vision_cache=False):
        # forward the GPT model itself
        vision = vision[:, :, None, :]
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if i % self.visual_ca_ds_ratio == 0:
                x = self.transformer.hv[i // self.visual_ca_ds_ratio](x, vision, use_cache=use_vision_cache)
        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        return x

    def cached_forward(self, x, vision, valid_mask=None):
        # forward the GPT model itself
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x, use_cache=True, valid_mask=valid_mask)
            if i % self.visual_ca_ds_ratio == 0:
                x = self.transformer.hv[i // self.visual_ca_ds_ratio](x, vision, use_cache=True)
        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        return x

    def clear_kv_cache(self):
        for block in self.transformer.h:
            block.clear_kv_cache()
        for block in self.transformer.hv:
            block.clear_kv_cache()