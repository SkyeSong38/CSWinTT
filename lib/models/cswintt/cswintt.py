from torch import nn

from lib.utils.misc import NestedTensor
from lib.models.cswintt import build_backbone,build_transformer_cs,build_box_head,MLP
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.image import *

class CSWinTT(nn.Module):
	""" This is the base class for Transformer Tracking """

	def __init__(self, backbone, transformer, box_head, num_queries,
	             aux_loss=False, head_type="CORNER", cls_head=None):
		""" Initializes the model.
		Parameters:
			backbone: torch module of the backbone to be used. See backbone.py
			transformer: torch module of the transformer architecture. See transformer.py
			num_queries: number of object queries.
			aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
		"""
		super().__init__()
		self.backbone = backbone
		self.transformer = transformer
		self.box_head = box_head
		self.num_queries = num_queries
		hidden_dim = transformer.d_model
		self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
		self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
		self.aux_loss = aux_loss
		self.head_type = head_type
		if head_type == "CORNER":
			self.feat_sz_s = int(box_head.feat_sz)
			self.feat_len_s = int(box_head.feat_sz ** 2)

		self.cls_head = cls_head

	def forward(self, input=None, feat_dict_list=None, mode="backbone", run_box_head=False, run_cls_head=False):
		if mode == "backbone":
			return self.forward_backbone(input)
		elif mode == "transformer":
			return self.forward_transformer(feat_dict_list, run_box_head=run_box_head, run_cls_head=run_cls_head)
		else:
			raise ValueError

	def forward_backbone(self, input: NestedTensor):
		"""The input type is NestedTensor, which consists of:
			   - tensor: batched images, of shape [batch_size x 3 x H x W]
			   - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
		"""
		assert isinstance(input, NestedTensor)
		# Forward the backbone
		output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
		# Adjust the shapes
		return self.adjust(output_back, pos)

	def forward_transformer(self, feat_dict_list, run_box_head=False, run_cls_head=False):
		if self.aux_loss:
			raise ValueError("Deep supervision is not supported.")
		# Forward the transformer encoder and decoder
		merge_feat, hs = self.transformer(feat_dict_list[0], feat_dict_list[1], feat_dict_list[2], self.query_embed.weight)
		# Forward the corner head
		out, outputs_coord = self.forward_head(merge_feat, hs, run_box_head=run_box_head, run_cls_head=run_cls_head)
		return out, outputs_coord, hs

	def forward_head(self, merge_feat, hs, run_box_head=False, run_cls_head=False):
		"""
		hs: output embeddings (1, B, N, C)
		memory: encoder embeddings (HW1+HW2, B, C)"""
		out_dict = {}
		if run_cls_head:
			# forward the classification head
			out_dict.update({'pred_logits': self.cls_head(hs)[-1]})
		if run_box_head:
			# forward the box prediction head
			out_dict_box, outputs_coord = self.forward_box_head(merge_feat, hs)
			# merge results
			out_dict.update(out_dict_box)
			return out_dict, outputs_coord
		else:
			return out_dict, None

	def forward_box_head(self, merge_feat, hs):
		"""
		hs: output embeddings (N, B, C)
		memory: encoder embeddings (HW1+HW2, B, C)"""
		if self.head_type == "CORNER":
			# ------------adjust shape------------
			hs = hs.permute(1, 2, 0)  # (B, C, N)
			merge_feat = merge_feat.permute(1, 0, 2) #(B, HW, C)
			att = torch.matmul(merge_feat, hs)  # (B, HW, N)
			opt = (merge_feat * att).permute(0, 2, 1).contiguous()  # (B, HW, C) --> (B, C, HW)
			bs, C, HW = opt.size()
			opt_feat = opt.view(bs, C, self.feat_sz_s, self.feat_sz_s)
			outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
			# ------------return data-----------
			Nq = 1
			outputs_coord_new = outputs_coord.view(bs, Nq, 4)
			out = {'pred_boxes': outputs_coord_new}
			return out, outputs_coord_new
		elif self.head_type == "MLP":
			# Forward the class and box head
			outputs_coord = self.box_head(hs).sigmoid()
			out = {'pred_boxes': outputs_coord[-1]}
			if self.aux_loss:
				out['aux_outputs'] = self._set_aux_loss(outputs_coord)
			return out, outputs_coord

	def adjust(self, output_back: list, pos_embed: list):
		"""
		"""
		src_feat, mask = output_back[-1].decompose()
		assert mask is not None
		# reduce channel
		feat = self.bottleneck(src_feat)  # (B, C, H, W)
		# adjust shapes
		feat = feat.permute(2, 3, 0, 1)  # HxWxBxC
		if pos_embed is not None and len(pos_embed) != 0:
			pos_embed = pos_embed[-1].permute(2, 3, 0, 1)  # HxWxBxC
		return {"feat": feat, "mask": mask, "pos": pos_embed}

	@torch.jit.unused
	def _set_aux_loss(self, outputs_coord):
		# this is a workaround to make torchscript happy, as torchscript
		# doesn't support dictionary with non-homogeneous values, such
		# as a dict having both a Tensor and a list.
		return [{'pred_boxes': b}
		        for b in outputs_coord[:-1]]

def build_cswintt(cfg):
	backbone = build_backbone(cfg)  # backbone and positional encoding are built together
	transformer = build_transformer_cs(cfg)
	box_head = build_box_head(cfg)
	cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
	model = CSWinTT(
		backbone,
		transformer,
		box_head,
		num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
		aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
		head_type=cfg.MODEL.HEAD_TYPE,
		cls_head=cls_head
	)

	return model
