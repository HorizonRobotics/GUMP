import math

import torch
from torch import nn
    
def get_sine_embedding_2d(x, y, heading, width, length, speed, n_embd):
    """
    get 2d sine embedding
    input: x : torch.Tensor, shape (1, ) x in world coordinate[-50m 50m]
           y : torch.Tensor, shape (1, ) y in world coordinate[-50m 50m]
    output: sine_embedding: torch.Tensor, shape (1, n_embd)
    """
    
    # Determine dtype and device from one of the non-None inputs
    for tensor in [x, y, heading, width, length, speed]:
        if tensor is not None:
            dtype = tensor.dtype
            device = tensor.device
            shape = tensor.shape
            break
            
    # Frequency for positional encoding
    d = n_embd // 12
    div_term = torch.exp(torch.arange(0., d, 1, dtype=dtype, device=device) * -(torch.log(torch.tensor(10000.0, dtype=dtype, device=device)) / d))
    
    x_embed = sin_cos_embed_or_zero(x, div_term, dtype, device, shape)
    y_embed = sin_cos_embed_or_zero(y, div_term, dtype, device, shape)
    heading_embed = sin_cos_embed_or_zero(heading, div_term, dtype, device, shape)
    width_embed = sin_cos_embed_or_zero(width, div_term, dtype, device, shape)
    length_embed = sin_cos_embed_or_zero(length, div_term, dtype, device, shape)
    speed_embed = sin_cos_embed_or_zero(speed, div_term, dtype, device, shape)
    
    # Concatenate the embeddings
    # try:
    sine_embedding = torch.cat((x_embed, y_embed, heading_embed, width_embed, length_embed, speed_embed), dim=-1)
    # except:
    #     import pdb; pdb.set_trace()
    if n_embd - d * 12 > 0:
        sine_embedding = torch.cat((sine_embedding, torch.zeros((*sine_embedding.shape[:-1], n_embd - d * 12), dtype=dtype, device=device)), dim=-1)
    
    return sine_embedding

def sin_cos_embed_or_zero(tensor, div_term, dtype, device, shape):
    if tensor is not None:
        return torch.cat((torch.sin(tensor * div_term), torch.cos(tensor * div_term)), dim=-1)
    else:
        return torch.zeros((*shape[:-1], len(div_term)*2), dtype=dtype, device=device)


def get_sine_embedding_kinetics_2d(x, y, heading, width, length, lon_speed, steering_angle, n_embd):
    """
    get 2d sine embedding
    input: x : torch.Tensor, shape (1, ) x in world coordinate[-50m 50m]
           y : torch.Tensor, shape (1, ) y in world coordinate[-50m 50m]
    output: sine_embedding: torch.Tensor, shape (1, n_embd)
    """
    
    # Determine dtype and device from one of the non-None inputs
    for tensor in [x, y, heading, width, length, lon_speed, steering_angle]:
        if tensor is not None:
            dtype = tensor.dtype
            device = tensor.device
            shape = tensor.shape
            break
            
    # Frequency for positional encoding
    d = n_embd // 10
    div_term = torch.exp(torch.arange(0., d, 1, dtype=dtype, device=device) * -(torch.log(torch.tensor(10000.0, dtype=dtype, device=device)) / d))
    
    x_embed = sin_cos_embed_or_zero(x, div_term, dtype, device, shape)
    y_embed = sin_cos_embed_or_zero(y, div_term, dtype, device, shape)
    heading_embed = sin_cos_embed_or_zero(heading, div_term, dtype, device, shape)
    width_embed = sin_cos_embed_or_zero(width, div_term, dtype, device, shape)
    length_embed = sin_cos_embed_or_zero(length, div_term, dtype, device, shape)

    lon_speed_embed = sin_cos_embed_or_zero(lon_speed, div_term, dtype, device, shape)
    steering_angle_embed = sin_cos_embed_or_zero(steering_angle, div_term, dtype, device, shape)
    
    # Concatenate the embeddings
    # try:
    sine_embedding = torch.cat((x_embed, y_embed, heading_embed, width_embed, length_embed, lon_speed_embed, steering_angle_embed), dim=-1)

    # except:
    #     import pdb; pdb.set_trace()
    # if n_embd - d * 10 > 0:
    #     sine_embedding = torch.cat((sine_embedding, torch.zeros((*sine_embedding.shape[:-1], n_embd - d * 10), dtype=dtype, device=device)), dim=-1)

    return sine_embedding