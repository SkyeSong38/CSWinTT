import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer_post(nn.Module):
	def __init__(self, d_model, num_layers, nhead=8, dim_feedforward=2048, dropout=0.1, ):
		super().__init__()
		decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
		self.layers = _get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = nn.LayerNorm(d_model)

	def forward(self, tgt, memory,
	            tgt_mask: Optional[Tensor] = None,
	            memory_mask: Optional[Tensor] = None,
	            tgt_key_padding_mask: Optional[Tensor] = None,
	            memory_key_padding_mask: Optional[Tensor] = None,
	            pos: Optional[Tensor] = None,
	            query_pos: Optional[Tensor] = None):
		output = tgt

		for layer in self.layers:
			output = layer(output, memory, tgt_mask=tgt_mask,
			               memory_mask=memory_mask,
			               tgt_key_padding_mask=tgt_key_padding_mask,
			               memory_key_padding_mask=memory_key_padding_mask,
			               pos=pos, query_pos=query_pos)

		if self.norm is not None:
			output = self.norm(output)

		return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, divide_norm=False):
		super().__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

		self.activation = nn.ReLU(inplace=True)

		self.divide_norm = divide_norm
		self.scale_factor = float(d_model // nhead) ** 0.5

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def forward(self, tgt, memory,
	            tgt_mask: Optional[Tensor] = None,
	            memory_mask: Optional[Tensor] = None,
	            tgt_key_padding_mask: Optional[Tensor] = None,
	            memory_key_padding_mask: Optional[Tensor] = None,
	            pos: Optional[Tensor] = None,
	            query_pos: Optional[Tensor] = None):
		# self-attention
		q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
		if self.divide_norm:
			q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
			k = k / torch.norm(k, dim=-1, keepdim=True)
		tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
		                      key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		# mutual attention
		queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
		if self.divide_norm:
			queries = queries / torch.norm(queries, dim=-1, keepdim=True) * self.scale_factor
			keys = keys / torch.norm(keys, dim=-1, keepdim=True)
		tgt2 = self.multihead_attn(query=queries,
		                           key=keys,
		                           value=memory, attn_mask=memory_mask,
		                           key_padding_mask=memory_key_padding_mask)[0]
		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_post():
	return Transformer_post()
