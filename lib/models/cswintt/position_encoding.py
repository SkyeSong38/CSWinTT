"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lib.utils.misc import NestedTensor


class PositionEmbeddingRelative(nn.Module):
    def __init__(self, window_size, num_heads):
        super(PositionEmbeddingRelative, self).__init__()
        self.window_size = (window_size, window_size)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias


class PositionEmbeddingRelative_(nn.Module):
    def __init__(self, window_size, L_win):
        super(PositionEmbeddingRelative_, self).__init__()
        self.window_size = window_size
        self.Len_all = L_win
        self.Num_win = L_win // window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.Num_win - 1))).cuda()  # 2*W-1 * 2*Nw-1

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.Num_win)
        coords_w = torch.arange(self.window_size).repeat(self.Num_win//window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, W*W*Nw
        coords_flatten = torch.stack([torch.flatten(coords[0, :, :self.window_size]),
                                      torch.flatten(coords[1, :self.window_size, :])]) # 2, W*W*Nw
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, W*W*Nw, W*W*Nw
        relative_coords[0] += self.Num_win - 1  # shift to start from 0
        relative_coords[1] += self.window_size - 1
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.Len_all, self.Len_all)  # Len_all,Len_all
        relative_position_bias = relative_position_bias.contiguous()  # Len_all,Len_all
        return relative_position_bias


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask  # (b,h,w)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # cumulative sum along axis 1 (h axis) --> (b, h, w)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # cumulative sum along axis 2 (w axis) --> (b, h, w)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 2pi * (y / sigma(y))
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 2pi * (x / sigma(x))

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # (0,1,2,...,d/2)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos  # (H,W,C) --> (C,H,W) --> (1,C,H,W) --> (B,C,H,W)


class PositionEmbeddingNone(nn.Module):
    """
    No positional encoding.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.n_dim = num_pos_feats * 2

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        b, _, h, w = x.size()
        return torch.zeros((b, self.n_dim, h, w), device=x.device)  # (B, C, H, W)


def build_position_encoding(cfg):
    N_steps = cfg.MODEL.HIDDEN_DIM // 2
    if cfg.MODEL.POSITION_EMBEDDING in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif cfg.MODEL.POSITION_EMBEDDING in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif cfg.MODEL.POSITION_EMBEDDING in ('None',):
        print("Not using positional encoding.")
        position_embedding = PositionEmbeddingNone(N_steps)
    else:
        raise ValueError(f"not supported {cfg.MODEL.POSITION_EMBEDDING}")

    return position_embedding
