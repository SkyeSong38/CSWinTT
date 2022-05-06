import torch
from torchvision.ops.boxes import box_area
import math
import numpy as np

def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(-1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
	     (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
	x1, y1, w, h = x.unbind(-1)
	b = [x1, y1, x1 + w, y1 + h]
	return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
	x1, y1, x2, y2 = x.unbind(-1)
	b = [x1, y1, x2 - x1, y2 - y1]
	return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
	x0, y0, x1, y1 = x.unbind(-1)
	b = [(x0 + x1) / 2, (y0 + y1) / 2,
	     (x1 - x0), (y1 - y0)]
	return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
	"""

	:param boxes1: (N, 4) (x1,y1,x2,y2)
	:param boxes2: (N, 4) (x1,y1,x2,y2)
	:return:
	"""
	area1 = box_area(boxes1)  # (N,)
	area2 = box_area(boxes2)  # (N,)

	lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
	rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

	wh = (rb - lt).clamp(min=0)  # (N,2)
	inter = wh[:, 0] * wh[:, 1]  # (N,)

	union = area1 + area2 - inter

	iou = inter / union
	return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
	"""
	Generalized IoU from https://giou.stanford.edu/

	The boxes should be in [x0, y0, x1, y1] format

	boxes1: (N, 4)
	boxes2: (N, 4)
	"""
	# degenerate boxes gives inf / nan results
	# so do an early check
	# try:
	assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
	assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
	iou, union = box_iou(boxes1, boxes2)  # (N,)

	lt = torch.min(boxes1[:, :2], boxes2[:, :2])
	rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

	wh = (rb - lt).clamp(min=0)  # (N,2)
	area = wh[:, 0] * wh[:, 1]  # (N,)

	return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
	"""

	:param boxes1: (N, 4) (x1,y1,x2,y2)
	:param boxes2: (N, 4) (x1,y1,x2,y2)
	:return:
	"""
	giou, iou = generalized_box_iou(boxes1, boxes2)
	return (1 - giou).mean(), iou


def clip_box(box: list, H, W, margin=0):
	x1, y1, w, h = box
	x2, y2 = x1 + w, y1 + h
	x1 = min(max(0, x1), W - margin)
	x2 = min(max(margin, x2), W)
	y1 = min(max(0, y1), H - margin)
	y2 = min(max(margin, y2), H)
	w = max(margin, x2 - x1)
	h = max(margin, y2 - y1)
	return [x1, y1, w, h]


def diou_loss(preds, bbox, reduction='mean'):
	'''
	https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
	:param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
	:param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
	:param eps: eps to avoid divide 0
	:param reduction: mean or sum
	:return: diou-loss
	'''
	iou, union = box_iou(preds, bbox)

	# inter_diag
	cxpreds = (preds[:, 2] + preds[:, 0]) / 2
	cypreds = (preds[:, 3] + preds[:, 1]) / 2

	cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
	cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

	inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

	# outer_diag
	ox1 = torch.min(preds[:, 0], bbox[:, 0])
	oy1 = torch.min(preds[:, 1], bbox[:, 1])
	ox2 = torch.max(preds[:, 2], bbox[:, 2])
	oy2 = torch.max(preds[:, 3], bbox[:, 3])

	outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

	diou = iou - inter_diag / outer_diag
	diou = torch.clamp(diou, min=-1.0, max=1.0)

	diou_loss = 1 - diou

	if reduction == 'mean':
		loss = torch.mean(diou_loss)
	elif reduction == 'sum':
		loss = torch.sum(diou_loss)
	else:
		raise NotImplementedError
	return loss, iou

def ciou_loss(preds, bbox, reduction='mean'):
	iou, union = box_iou(preds, bbox)

	# inter_diag
	cxpreds = (preds[:, 2] + preds[:, 0]) / 2
	cypreds = (preds[:, 3] + preds[:, 1]) / 2

	cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
	cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

	inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

	# outer_diag
	ox1 = torch.min(preds[:, 0], bbox[:, 0])
	oy1 = torch.min(preds[:, 1], bbox[:, 1])
	ox2 = torch.max(preds[:, 2], bbox[:, 2])
	oy2 = torch.max(preds[:, 3], bbox[:, 3])

	outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

	diou = iou - inter_diag / outer_diag

	# calculate v,alpha
	wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
	hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
	wpreds = preds[:, 2] - preds[:, 0] + 1.0
	hpreds = preds[:, 3] - preds[:, 1] + 1.0
	v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
	alpha = v / (1 - iou + v)
	ciou = diou - alpha * v
	ciou = torch.clamp(ciou, min=-1.0, max=1.0)

	ciou_loss = 1 - ciou
	if reduction == 'mean':
		loss = torch.mean(ciou_loss)
	elif reduction == 'sum':
		loss = torch.sum(ciou_loss)
	else:
		raise NotImplementedError
	return loss, iou
