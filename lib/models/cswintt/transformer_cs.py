from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import trunc_normal_
from lib.models.cswintt.transformer_post import TransformerDecoderLayer
from lib.utils.image import *


def check_inf(tensor):
	return torch.isinf(tensor.detach()).any()

def check_nan(tensor):
	return torch.isnan(tensor.detach()).any()

def check_valid(tensor, type_name):
	if check_inf(tensor):
		print("%s is inf." % type_name)
	if check_nan(tensor):
		print("%s is nan" % type_name)

class Transformer_CS(nn.Module):

	def __init__(self, search_size, template_size, d_model=256, nhead=8, d_feedforward=2048, stack_num=6, dropout=0.1):
		super().__init__()

		self.h1, self.w1 = search_size
		self.h2, self.w2 = template_size
		self.d_model = d_model
		self.nhead = nhead
		self.d_feedforward = d_feedforward

		self.L_all = self.h1 * self.w1 + 2 * self.h2 * self.w2
		self.window_list = [1, 2, 4]
		for window_size in self.window_list:
			L = self.L_all // window_size // window_size
			self.register_parameter("rel_bias_table" + str(window_size), nn.Parameter(
				torch.zeros((2 * window_size - 1) * (self.L_all // window_size + L // window_size - 1),
				            requires_grad=True).cuda().requires_grad_()))
			trunc_normal_(self.__getattr__("rel_bias_table" + str(window_size)), std=.02)

		blocks = []
		for i in range(stack_num):
			window_shift = False if i // 2 == 0 else True
			window_size = 0
			blocks.append(
				TransformerLayer(self.d_model, self.nhead, search_size, template_size, window_size, window_shift,
				                 self.d_feedforward, dropout))
		self.transformer = nn.ModuleList(blocks)

		decoder_blocks = []
		for i in range(stack_num):
			decoder_blocks.append(
				TransformerDecoderLayer(self.d_model, self.nhead, self.d_feedforward, dropout))
		self.transformer_post = nn.ModuleList(decoder_blocks)
		self.post_norm = nn.LayerNorm(self.d_model)

		self._reset_parameters()

	def get_relative_position_index(self, window_size):
		Num_win = self.L_all // window_size
		L = self.L_all // window_size // window_size
		# get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(Num_win)
		coords_w = torch.arange(window_size).repeat(Num_win // window_size)
		coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, W*W*Nw
		coords_flatten = torch.stack([torch.flatten(coords[0, :, :window_size]),
		                              torch.flatten(coords[1, :window_size, :])])  # 2, W*W*Nw
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, W*W*Nw, W*W*Nw
		relative_coords[0] += Num_win - 1  # shift to start from 0
		relative_coords[1] += window_size - 1
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()
		relative_coords[:, :, 0] *= 2 * window_size - 1
		relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
		relative_position_index = relative_position_index[:L]
		return relative_position_index

	def get_relative_position(self):
		relative_position = {}
		for window_size in self.window_list:
			rel_bias_table = self.__getattr__('rel_bias_table' + str(window_size))
			# rel_bias_table = self.rel_bias_table.get(window_size)
			rel_index = self.get_relative_position_index(window_size)
			relative_position_bias = rel_bias_table[rel_index.view(-1)].view(self.L_all // window_size // window_size,
			                                                                 self.L_all)
			relative_position_bias = relative_position_bias.contiguous()
			relative_position.update({window_size: relative_position_bias})
		return relative_position

	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, template_1, template_2, search, query_embed):
		# def forward(self, merge_feat, merge_mask, query_embed, merge_pos, mode="all", return_encoder_output=False):
		"""
		:param search_feat: (h1, w1, b, c)
		:param template_feat: (h2, w2, b, c)
		:param search_mask: (b, h1, w1)
		:param template_mask: (b, h2, w2)
		:param search_pos: (h1, w1, b, c)
		:param template_pos: (h2, w2, b,c)
		:return: (h2, w2, b, c), (h1, w1, b, c)
		"""

		# -------------------merge---------------------
		b = search["feat"].size(2)
		# (L_all, b , c) , L_all = h2*w2 + h2*w2 + h1*w1
		merge_feat = torch.cat(
			[template_1["feat"].flatten(0, 1), template_2["feat"].flatten(0, 1), search["feat"].flatten(0, 1)],
			dim=0)

		# (b, L_all)
		merge_mask = torch.cat(
			[template_1["mask"].flatten(1, 2), template_2["mask"].flatten(1, 2), search["mask"].flatten(1, 2)],
			dim=1)

		# (L_all, b, c)
		merge_pos = torch.cat(
			[template_1["pos"].flatten(0, 1), template_2["pos"].flatten(0, 1), search["pos"].flatten(0, 1)],
			dim=0)
		# -------------------transformer---------------------
		first_L = self.h2 * self.w2

		rel_pos = self.get_relative_position()
		for layer in self.transformer:
			merge_feat = layer(merge_feat=merge_feat, merge_mask=merge_mask, rel_pos=rel_pos, merge_pos=merge_pos)

		# merge_feat: (L_all, b, c)
		search_feat = merge_feat[first_L * 2:]
		# -------------------transformer_post---------------------
		query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
		tgt = torch.zeros_like(query_embed)
		output = tgt
		for layer in self.transformer_post:
			output = layer(output, merge_feat, memory_key_padding_mask=merge_mask, pos=None, query_pos=query_embed)
		hs = self.post_norm(output)

		return search_feat, hs


class TransformerLayer(nn.Module):
	def __init__(self, d_model, nhead, search_size, template_size, window_size, window_shift, dim_feedforward=2048,
	             dropout=0.1):
		super().__init__()
		self.h1, self.w1 = search_size
		self.h2, self.w2 = template_size
		if window_size == 0:
			self.window_size = [1, 2, 4, 8, 1, 2, 4, 8]
			# self.window_size = [1, 2, 2, 4, 1, 2, 2, 4]
			# self.window_size = [1, 1, 1, 1, 1, 1, 1, 1]
			self.window_shift = None
		else:
			self.window_size = [window_size, window_size, window_size, window_size, window_size, window_size,
			                    window_size,
			                    window_size]
			self.window_shift = window_shift

		self.attn = Attention(self.window_size, dropout)

		self.qkv_embedding = nn.Linear(d_model, d_model * 3)
		self.output_linear = nn.Linear(d_model, d_model)
		self.nhead = nhead
		head_dim = d_model // nhead
		self.scale = head_dim ** -0.5

		# Implementation of Feedforward model
		self.feedforward = nn.Sequential(
			nn.Linear(d_model, dim_feedforward),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(dim_feedforward, d_model),
			nn.Dropout(dropout)
		)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

		self.scale_factor = float(d_model // nhead) ** 0.5

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def _window_partition(self, merge_feat, window_size, shift_size=0):
		"""
		:param merge_feat: (b, L_all, c)
		:return: (b, L, window_size*window_size*c)
		"""
		if window_size <= 1:
			return merge_feat
		b, L_all, c = merge_feat.size()
		first_L = self.h2 * self.w2
		window_size_h = window_size_w = window_size
		c_win = window_size_h * window_size_w * c

		t1 = merge_feat[:, :first_L, :]  # (b, first_L, c)
		t1 = t1.contiguous().view(b, self.h2 // window_size_h, window_size_h, self.w2 // window_size_w, window_size_w,
		                          c)
		t1 = t1.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, c_win)

		t2 = merge_feat[:, first_L:2 * first_L, :]  # (b, first_L, c)
		t2 = t2.contiguous().view(b, self.h2 // window_size_h, window_size_h, self.w2 // window_size_w, window_size_w,
		                          c)
		t2 = t2.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, c_win)

		s = merge_feat[:, first_L * 2:, :]
		s_shift = s.contiguous().view(b, self.h1, self.w1, c)
		if shift_size > 0:
			s_shift = torch.roll(s_shift, shifts=(-shift_size, -shift_size), dims=(1, 2))

		s = s_shift.contiguous().view(b, self.h1 // window_size_h, window_size_h, self.w1 // window_size_w,
		                              window_size_w, c)
		s = s.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, c_win)

		return torch.cat([t1, t2, s], dim=1)

	def _window_reverse(self, merge_feat, window_size, shift_size):
		"""
		:param merge_feat: (b, L, window_size*window_size*c)
		:return: (b, L_all, c)
		"""
		if window_size <= 1:
			return merge_feat
		window_size_h = window_size_w = window_size

		b, L, c_win = merge_feat.size()
		c = c_win // window_size_h // window_size_w
		first_L = (self.h2 // window_size_h) * (self.w2 // window_size_w)
		last_L = (self.h1 // window_size_h) * (self.w1 // window_size_w)

		t1 = merge_feat[:, :first_L, :]  # (b, first_L, window_size*window_size*c)
		t1 = t1.view(b, self.h2 // window_size_h,
		             self.w2 // window_size_w, window_size_h, window_size_w, c)
		t1 = t1.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, c)

		t2 = merge_feat[:, first_L:2 * first_L, :]  # (b, first_L, window_size*window_size*c)
		t2 = t2.view(b, self.h2 // window_size_h,
		             self.w2 // window_size_w, window_size_h, window_size_w, c)
		t2 = t2.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, c)

		s = merge_feat[:, first_L * 2:, :]  # (b, last_L, window_size*window_size*c)
		s = s.view(b, self.h1 // window_size_h,
		           self.w1 // window_size_w, window_size_h, window_size_w, c)
		s_shift = s.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, self.h1, self.w1, c)
		if shift_size > 0:
			s_shift = torch.roll(s_shift, shifts=(shift_size, shift_size), dims=(1, 2))
		s = s_shift.view(b, -1, c)
		return torch.cat([t1, t2, s], dim=1)

	def forward(self, merge_feat, merge_mask, rel_pos=None, merge_pos=None):
		# merge_feat = (L,b,c)
		# start_time = time.time()
		# -------------------Multi-head attention---------------------
		_query, _key, _value = self.qkv_embedding(merge_feat).chunk(3, dim=-1)

		# print("embedding_time: " + str(time.time() - start_time))
		_query = _query * self.scale
		output = []
		for index, (window_size, query, key, value) in enumerate(zip(
				self.window_size,
				torch.chunk(_query, self.nhead, dim=-1),
				torch.chunk(_key, self.nhead, dim=-1),
				torch.chunk(_value, self.nhead, dim=-1))
		):
			query = query.transpose(0, 1)
			key = key.transpose(0, 1)
			value = value.transpose(0, 1)

			if self.window_shift or (self.window_shift is None and index >= self.nhead // 2):
				shift_size = window_size // 2
			else:
				shift_size = 0

			# -------------------window partition---------------------
			# (b, L_all, c)  ---> (b, L, win_h * win_w * c)
			query = self._window_partition(query, window_size, shift_size)
			key = self._window_partition(key, window_size, shift_size)
			value = self._window_partition(value, window_size, shift_size)

			mask = self._window_partition(merge_mask.unsqueeze(-1), window_size)
			mask = mask.to(torch.float16).mean(-1) >= 1

			# -------------------attention---------------------
			attn, _ = self.attn(query, key, value, window_size, mask, rel_pos.get(window_size))

			# -------------------window_reverse---------------------
			attn = self._window_reverse(attn, window_size, shift_size)
			output.append(attn.transpose(0, 1))

		# print("window_" + str(window_size) + "_time: " + str(time.time() - start_time))
		output = torch.cat(output, -1)

		output = self.output_linear(output)
		# -------------------FeedForward---------------------

		merge_feat = merge_feat + self.dropout(output)
		merge_feat = self.norm1(merge_feat)
		merge_feat = merge_feat + self.feedforward(merge_feat)
		merge_feat = self.norm2(merge_feat)

		return merge_feat


class Attention(nn.Module):
	"""
	Compute 'Scaled Dot Product Attention
	"""

	def __init__(self, window_size_list, p=0.1):
		super(Attention, self).__init__()
		self.dropout = nn.Dropout(p=p)

		self.x_mesh = {}
		self.y_mesh = {}
		# self.matrix_index_query = {}
		# self.matrix_index_key = {}
		self.sr_attn_mask = {}

		L_win = 704

		for win_size in window_size_list:
			if win_size == 1 or self.sr_attn_mask.get(win_size) is not None:
				continue
			# spatially regularized attention mask
			L = L_win // win_size // win_size
			attn_mask_w = torch.tensor([x if x > 0.5 else 1 - x for x in np.arange(0, 1, 1 / win_size)],
			                           dtype=torch.float16).view(1, win_size)
			attn_mask = attn_mask_w.T * attn_mask_w - 1.0
			attn_mask = attn_mask.unsqueeze(0).repeat(L, 1, 1).view(L_win).cuda()
			self.sr_attn_mask.update({win_size: attn_mask.unsqueeze(0).unsqueeze(0)})

			# index for cyclic shift
			i, j = torch.nonzero(torch.ones((win_size, win_size)), as_tuple=False).T
			index = j - i
			x, y = torch.meshgrid(index, index)
			self.x_mesh.update({win_size: x})
			self.y_mesh.update({win_size: y})

	# matrix_index = torch.tensor(range(win_size * win_size)).view(win_size, win_size)
	# self.matrix_index_query.update({win_size: matrix_index.view(win_size * win_size).repeat(
	# 	win_size * win_size).view(win_size * win_size, win_size * win_size)})
	# self.matrix_index_key.update({win_size: matrix_index[x, y].view(win_size, win_size, win_size,
	#                                                                 win_size).permute(0, 2, 1,
	#                                                                                   3).contiguous().view(
	# 	win_size * win_size, win_size * win_size)})

	def forward(self, query, key, value, window_size, padding_mask=None, rel_pos=None, use_attn_mask=True):
		b, L, c_win = query.size()
		c = c_win // window_size // window_size
		L_win = L * window_size * window_size

		if window_size == 1:
			_attn_score = torch.matmul(query, key.transpose(-2, -1))
			if padding_mask is not None:
				assert padding_mask.size() == key.size()[0:2]
				_attn_score = _attn_score.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
		else:
			# q,k,v =  (b, L, win_h * win_w * c)
			# mask = (b, L)
			query = query.view(b, L, window_size, window_size, c)
			key = key.view(b, L, window_size, window_size, c)
			value = value.view(b, L, window_size, window_size, c)
			# -------original calculation-----------
			# _attn = torch.zeros(b, L, L, window_size * window_size, window_size * window_size).cuda()
			# for q_i in range(0, window_size):
			# 	for q_j in range(0, window_size):
			# 		_query = query.roll(shifts=(q_i, q_j), dims=(2, 3))
			# 		for k_i in range(0, window_size):
			# 			for k_j in range(0, window_size):
			# 				_key = key.roll(shifts=(k_i, k_j), dims=(2, 3))
			# 				_attn[:, :, :, q_i * window_size + q_j, k_i * window_size + k_j] = \
			# 					torch.matmul(_query.view(b, L, c_win), _key.view(b, L, c_win).transpose(-2, -1))
			# ----------------------

			# -------speed calculation-----------
			key = key[:, :, self.x_mesh.get(window_size), self.y_mesh.get(window_size), :].view(b, L, window_size,
			                                                                                    window_size,
			                                                                                    window_size,
			                                                                                    window_size, c). \
				permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(b, L_win, c_win)  # shift -> (b, L_win, c_win)
			_attn_score = torch.matmul(query.view(b, L, c_win), key.transpose(-2, -1))

			value = value[:, :, self.x_mesh.get(window_size), self.y_mesh.get(window_size), :].view(b, L, window_size,
			                                                                                        window_size,
			                                                                                        window_size,
			                                                                                        window_size, c). \
				permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(b, L_win, c_win)  # shift -> (b, L_win, c_win)

			if use_attn_mask:
				_attn_score = _attn_score + self.sr_attn_mask.get(window_size)

			_attn_score = _attn_score / (window_size * window_size)

			if padding_mask is not None:
				assert padding_mask.size()[0] == b
				assert padding_mask.size()[1] == L
				_attn_score = _attn_score.view(b, L, L, window_size, window_size)
				_attn_score = _attn_score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1),
				                                      float("-inf"))
				_attn_score = _attn_score.view(b, L, L_win)

		if rel_pos is not None:
			_attn_score = _attn_score + rel_pos.unsqueeze(0)

		p_attn = F.softmax(_attn_score, dim=-1)
		p_attn = self.dropout(p_attn)
		p_val = torch.matmul(p_attn, value)
		return p_val, p_attn

def build_transformer_cs(cfg):
	search_size = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
	template_size = cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.STRIDE
	return Transformer_CS([search_size, search_size], [template_size, template_size], d_model=cfg.MODEL.HIDDEN_DIM,
	                      nhead=cfg.MODEL.TRANSFORMER.NHEADS,
	                      d_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
	                      stack_num=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
	                      dropout=cfg.MODEL.TRANSFORMER.DROPOUT)