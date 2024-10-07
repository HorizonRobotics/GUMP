# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import nn

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_temporal_layer: int = 2
    n_spatial_layer: int = 2
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CrossAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        vision: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_vision: torch.Tensor,
        is_causal: bool,
        pos_embeddings_agent: torch.Tensor,
        pos_embeddings_vision: torch.Tensor,
    ):  
        bsz, seqlen, _ = x.shape
        bsz, vseqlen, _ = vision.shape

        if pos_embeddings_agent is not None:
            x = x + pos_embeddings_agent
        if pos_embeddings_vision is not None:
            vision = vision + pos_embeddings_vision

        xq, xk, xv = self.wq(x), self.wk(vision), self.wv(vision)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, vseqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, vseqlen, self.n_local_heads, self.head_dim)
        # xq, _ = apply_rotary_emb(xq, xq, freqs_cis=freqs_cis)
        # _, xk = apply_rotary_emb(xk, xk, freqs_cis=freqs_cis_vision)

        # (bs, n_local_heads, seqlen / vseqlen, head_dim)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        query = xq.transpose(1, 2)
        
        output = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=None, is_causal=is_causal)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        is_causal: bool,
        spatial_mask: torch.Tensor = None,
        pos_embeddings_agent: torch.Tensor = None,
    ):  
        if pos_embeddings_agent is not None:
            xqk = x + pos_embeddings_agent
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(xqk), self.wk(xqk), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        query = xq.transpose(1, 2)

        if spatial_mask is not None:
            spatial_mask = spatial_mask.view(bsz, 1, seqlen, seqlen)
            spatial_mask = spatial_mask.repeat(1, self.n_local_heads, 1, 1)
            spatial_mask = torch.where(spatial_mask, torch.tensor(-10000), torch.tensor(0.0))
        output = torch.nn.functional.scaled_dot_product_attention(query, keys, values, is_causal=is_causal, attn_mask=spatial_mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TemporalTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attention = SelfAttention(args)
        # self.cross_attention = CrossAttention(args)
        self.self_feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        # self.cross_feed_forward = FeedForward(
        #     dim=args.dim,
        #     hidden_dim=4 * args.dim,
        #     multiple_of=args.multiple_of,
        #     ffn_dim_multiplier=args.ffn_dim_multiplier,
        # )
        self.layer_id = layer_id
        self.self_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.cross_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        vision: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_vision: torch.Tensor,
        pos_embeddings_agent: torch.Tensor,
        pos_embeddings_vision: torch.Tensor,
    ):  
        bsz, frame_len, agent_len, _ = x.shape
        bsz, visual_seqlen, _ = vision.shape

        # Temporal attention is along the frame length dimention
        x = x.permute(0, 2, 1, 3)
        pos_embeddings_agent = pos_embeddings_agent.permute(0, 2, 1, 3)
        
        x = x.contiguous().view(bsz * agent_len, frame_len, -1)
        pos_embeddings_agent = pos_embeddings_agent.contiguous().view(bsz * agent_len, frame_len, -1)
        # vision = vision.repeat(agent_len, 1, 1)
        # pos_embeddings_vision = pos_embeddings_vision.repeat(agent_len, 1, 1)
        
        h = x + self.self_attention(self.self_attention_norm(x), freqs_cis, is_causal=True, pos_embeddings_agent=pos_embeddings_agent)
        out = h + self.self_feed_forward(self.self_ffn_norm(h))
        
        # h = h + self.cross_attention(self.cross_attention_norm(h), vision, freqs_cis, freqs_cis_vision, is_causal=False, pos_embeddings_agent=pos_embeddings_agent, pos_embeddings_vision=pos_embeddings_vision)
        # out = h + self.cross_feed_forward(self.cross_ffn_norm(h))

        out = out.contiguous().view(bsz, agent_len, frame_len, -1)
        out = out.permute(0, 2, 1, 3)

        return out

class SpatialTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attention = SelfAttention(args)
        self.self_feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.cross_attention = CrossAttention(args)
        self.cross_feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.self_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cross_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        vision: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_vision: torch.Tensor,
        pos_embeddings_agent: torch.Tensor,
        pos_embeddings_vision: torch.Tensor,
        spatial_mask: torch.Tensor,
    ):
        bsz, frame_len, agent_len, _ = x.shape
        bsz, visual_seqlen, _ = vision.shape

        # Spatial attention is along the agent length dimention
        x = x.contiguous().view(bsz * frame_len, agent_len, -1)
        pos_embeddings_agent = pos_embeddings_agent.contiguous().view(bsz * frame_len, agent_len, -1)
        vision = vision.repeat(frame_len, 1, 1)
        pos_embeddings_vision = pos_embeddings_vision.repeat(frame_len, 1, 1)

        h = x + self.self_attention(self.self_attention_norm(x), None, is_causal=False, spatial_mask=spatial_mask, pos_embeddings_agent=pos_embeddings_agent)
        h = h + self.self_feed_forward(self.self_ffn_norm(h))

        h = h + self.cross_attention(self.cross_attention_norm(h), vision, freqs_cis, freqs_cis_vision, is_causal=False, pos_embeddings_agent=pos_embeddings_agent, pos_embeddings_vision=pos_embeddings_vision)
        out = h + self.cross_feed_forward(self.cross_ffn_norm(h))
        out = out.contiguous().view(bsz, frame_len, agent_len, -1)        
        return out


class Kinetics_GUMP(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Hard code for two different transformer layers
        self.n_temporal_layer = params.n_temporal_layer
        self.n_spatial_layer = params.n_spatial_layer

        self.temporal_layer = torch.nn.ModuleList()
        for layer_id in range(self.n_temporal_layer):
            self.temporal_layer.append(TemporalTransformerBlock(layer_id, params))
        
        self.spatial_layer = torch.nn.ModuleList()
        for layer_id in range(self.n_temporal_layer, self.n_temporal_layer + self.n_spatial_layer):
            self.spatial_layer.append(SpatialTransformerBlock(layer_id, params))
        assert self.n_temporal_layer == self.n_spatial_layer, "Temporal and Spatial layers must be equal."

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )


    def forward_train(self, agent_token: torch.Tensor, visual: torch.Tensor, start_pos: int = 0, valid_mask: torch.Tensor = None, pos_embeddings_agent: torch.Tensor = None, pos_embeddings_vision: torch.Tensor = None):
        # valid_mask (bs, frame_len, agent_len)
        bsz, frame_len, agent_len, _ = agent_token.shape
        bsz, visual_seqlen, _ = visual.shape

        self.freqs_cis = self.freqs_cis.to(agent_token.device)
        freqs_cis_frame = self.freqs_cis[:frame_len]
        freqs_cis_agent = self.freqs_cis[:agent_len]
        freqs_cis_vision = self.freqs_cis[:visual_seqlen]
        
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool).to(agent_token.device)
        spatial_mask = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)

        h = agent_token
        for i, layer in enumerate(self.temporal_layer):
            h = self.temporal_layer[i](h, visual, freqs_cis_frame, freqs_cis_vision, pos_embeddings_agent, pos_embeddings_vision)
            h = self.spatial_layer[i](h, visual, freqs_cis_agent, freqs_cis_vision, pos_embeddings_agent, pos_embeddings_vision, spatial_mask)
            
        h = self.norm(h)
        output = h
        return output
    
    @torch.inference_mode()
    def forward_inference(self, agent_token: torch.Tensor, visual: torch.Tensor, start_pos: int = 0, valid_mask: torch.Tensor = None, pos_embeddings_agent: torch.Tensor = None, pos_embeddings_vision: torch.Tensor = None):
        bsz, frame_len, agent_len, _ = agent_token.shape
        bsz, visual_seqlen, _ = visual.shape

        self.freqs_cis = self.freqs_cis.to(agent_token.device)
        freqs_cis_frame = self.freqs_cis[:frame_len]
        freqs_cis_agent = self.freqs_cis[:agent_len]
        freqs_cis_vision = self.freqs_cis[:visual_seqlen]

        valid_mask = torch.tensor(valid_mask, dtype=torch.bool).to(agent_token.device)
        spatial_mask = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)

        h = agent_token
        for i, layer in enumerate(self.temporal_layer):
            # h: torch.Size([64, 3, 34, 256]), batch_size, frame_len, agent_len, dim
            # visual: torch.Size([64, 169, 256]), batch_size, visual_seqlen, dim
            # freqs_cis_frame: torch.Size([3, 32]), frame_len, dim / n_heads
            # freqs_cis_vision: torch.Size([169, 32]), visual_seqlen, dim / n_heads
            # freqs_cis_agent: torch.Size([34, 32]), agent_len, dim / n_heads
            # spatial_mask: torch.Size([64, 3, 34, 34]), batch_size, frame_len, agent_len, agent_len

            h = self.temporal_layer[i](h, visual, freqs_cis_frame, freqs_cis_vision, pos_embeddings_agent, pos_embeddings_vision)
            h = self.spatial_layer[i](h, visual, freqs_cis_agent, freqs_cis_vision, pos_embeddings_agent, pos_embeddings_vision, spatial_mask)

        h = self.norm(h)
        output = h
        return output