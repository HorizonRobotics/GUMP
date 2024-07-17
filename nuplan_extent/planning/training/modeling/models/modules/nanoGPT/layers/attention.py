
from einops import rearrange, repeat
from einops_exts import rearrange_many
import math

import torch
from torch import einsum, nn
from torch.nn import functional as F
import loralib as lora
# from torch.nn.attention.bias import causal_lower_right, causal_upper_left
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.layers.block import LayerNorm, FeedForward, MLP

def scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(
            L,
            S,
            dtype=torch.bool,
            device=query.device).tril(
            diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # self.c_attn = lora.Linear(config.n_embd, 3 * config.n_embd, r=8, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # self.c_proj = lora.Linear(config.n_embd, config.n_embd, r=8, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Initialize empty key and value caches
        self.key_cache = []
        self.value_cache = []
        self.query_cache = []
        self.q_start_index = []
        self.block_size = config.block_size

    def clear_kv_cache(self):
        """Clears the key and value caches."""
        self.key_cache = []
        self.value_cache = []
        self.query_cache = []
        self.q_start_index = []

    def forward(self, x, use_cache=False, valid_mask=None):
        B, T, C = x.size()

        # Compute query, key, and value for all heads, reshape for multi-head attention
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if use_cache:
            # Concatenate cached keys and values with current batch, if cache is not empty
            if valid_mask is not None:
                for i in range(B):
                    if len(self.key_cache) == B:
                        self.q_start_index[i] = self.key_cache[i].shape[1]
                        self.key_cache[i] = torch.cat([self.key_cache[i], k[i, :, valid_mask[i]]], dim=1) 
                        self.value_cache[i] = torch.cat([self.value_cache[i], v[i, :, valid_mask[i]]], dim=1) 
                        self.query_cache[i] = torch.cat([self.query_cache[i], q[i, :, valid_mask[i]]], dim=1)
                        
                        # sliding window clip to block size
                        for i in range(B):
                            if self.key_cache[i].shape[1] > self.block_size - T:
                                offset = self.key_cache[i].shape[1] - self.block_size + T
                                self.key_cache[i] = self.key_cache[i][:, -self.block_size + T:]
                                self.value_cache[i] = self.value_cache[i][:, -self.block_size + T:]
                                self.query_cache[i] = self.query_cache[i][:, -self.block_size + T:]
                                self.q_start_index[i] -= offset 

                    else:
                        self.key_cache.append(k[i, :, valid_mask[i]])
                        self.value_cache.append(v[i, :, valid_mask[i]])
                        self.query_cache.append(q[i, :, valid_mask[i]])
                        self.q_start_index.append(0)

                max_len = max([self.q_start_index[i] + T for i in range(B)])
                k = torch.stack([F.pad(self.key_cache[i], (0, 0, 0, max_len - self.key_cache[i].shape[1])) for i in range(B)], dim=0)
                v = torch.stack([F.pad(self.value_cache[i], (0, 0, 0, max_len - self.value_cache[i].shape[1])) for i in range(B)], dim=0)
                q = torch.stack([F.pad(self.query_cache[i], (0, 0, 0, max_len - self.query_cache[i].shape[1])) for i in range(B)], dim=0)
            else:
                k =[torch.cat([self.key_cache[i], k[i]], dim=1) for i in range(B)]
                v = [torch.cat([self.value_cache[i], v[i]], dim=1) for i in range(B)]
                q = [torch.cat([self.query_cache[i], q[i]], dim=1) for i in range(B)]

                self.q_start_index = [self.key_cache[i].shape[1] for i in range(B)]
                max_len = max([self.q_start_index[i] + T for i in range(B)])
                k =  torch.stack([F.pad(k[i], (0, 0, 0, max_len - k[i].shape[1])) for i in range(B)], dim=0)
                v =  torch.stack([F.pad(v[i], (0, 0, 0, max_len - v[i].shape[1])) for i in range(B)], dim=0)
                q =  torch.stack([F.pad(q[i], (0, 0, 0, max_len - q[i].shape[1])) for i in range(B)], dim=0)
        # Calculate scaled dot product attention or use flash attention if available
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            Lk = k.size(-2)  # Total sequence length including cache
            # Create a causal mask with KV cache size taken into account
            attn_mask = torch.tril(torch.ones(T, Lk, device=x.device, dtype=torch.bool), diagonal=Lk - T)
            y = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        
        if use_cache:
            y = torch.stack([y[i, :, self.q_start_index[i]:self.q_start_index[i]+T] for i in range(B)], dim=0)
        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # self.to_q = lora.Linear(dim, inner_dim, r=8, bias=False)
        # self.to_kv = lora.Linear(dim, inner_dim * 2, r=8, bias=False)
        # self.to_out = lora.Linear(inner_dim, dim, r=8, bias=False)
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Initialize empty key and value caches
        self.key_cache =  torch.empty(0)
        self.value_cache = torch.empty(0)

    def clear_kv_cache(self):
        """Clears the key and value caches."""
        self.key_cache = torch.empty(0)
        self.value_cache = torch.empty(0)

    def forward(
        self,
        x,
        media,
        use_cache=False,
    ):
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        if use_cache and self.key_cache.size(0) != 0:
            # Concatenate cached keys and values with current batch, if cache is not empty
            k = self.key_cache 
            v = self.value_cache
        else:
            media = rearrange(media, 'b t n d -> b (t n) d')
            k, v = self.to_kv(media).chunk(2, dim=-1)
            if use_cache:
                # Update the cache
                self.key_cache = k.detach()
                self.value_cache = v.detach()

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        # Use the PyTorch built-in function for scaled dot product attention
        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask=None)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads
        )
        self.attn_gate = nn.Parameter(torch.randn(1)*0.3)
        self.ff = FeedForward(dim, mult=ff_mult)

    def forward(
        self,
        x,
        media,
        query_pos=None,
        use_cache=False,
    ):
        x = (
            self.attn(
                x if query_pos is None else (x + query_pos),
                media,
                use_cache=use_cache
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) + x
        return x

    def clear_kv_cache(self):
        self.attn.clear_kv_cache()


class CausalSelfAttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False, valid_mask=None):
        x = x + self.attn(self.ln_1(x), use_cache=use_cache, valid_mask=valid_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
    def clear_kv_cache(self):
        self.attn.clear_kv_cache()
