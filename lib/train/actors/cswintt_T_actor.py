from . import CSWinTTActor
from lib.utils.image import *
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou

class CSWinTT_T_Actor(CSWinTTActor):
	""" Actor for training the STARK-S and STARK-ST(Stage1)"""

	def __init__(self, net, objective, loss_weight, settings):
		super().__init__(net, objective, loss_weight, settings)
		self.loss_weight = loss_weight
		self.settings = settings
		self.bs = self.settings.batchsize  # batch size

	def __call__(self, data):
		"""
		args:
			data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
			template_images: (N_t, batch, 3, H, W)
			search_images: (N_s, batch, 3, H, W)
		returns:
			loss    - the training loss
			status  -  dict containing detailed losses
		"""
		# forward pass
		out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=True)

		# process the groundtruth
		gt_labels = data['label'].view(-1)
		gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

		# draw_image(data['search_images'][0,0,:].permute(1,2,0).detach().cpu().numpy()*255+128, gt_bboxes[0,0,:].detach().cpu().numpy() * 384)
		# compute losses
		loss, status = self.compute_losses(out_dict, gt_bboxes, gt_labels)

		return loss, status

	def compute_losses(self, pred_dict, gt_bboxes, gt_labels , return_status=True):
		pred_boxes = pred_dict['pred_boxes']
		if torch.isnan(pred_boxes).any():
			raise ValueError("Network outputs is NAN! Stop Training")
		pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
		gt_boxes_vec = box_xywh_to_xyxy(gt_bboxes).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

		visiable_index = []
		for index, value in enumerate(gt_labels.view(-1)):
			if value == 0:
				visiable_index.append(index)

		pred_boxes_vec[visiable_index,:] = torch.zeros(4).cuda()
		gt_boxes_vec[visiable_index, :] = torch.zeros(4).cuda()
		try:
			ious = generalized_box_iou(pred_boxes_vec,gt_boxes_vec)[1]
		except:
			ious = None

		if ious is not None:
			remove_index = []
			for index, value in enumerate(ious):
				if torch.isnan(value) or value <= 0.5:
					remove_index.append(index)
			gt_labels[remove_index] = 0.

		loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), gt_labels)

		if return_status:
			# status for log
			status = {
				"cls_loss": loss.item()}
			return loss, status
		else:
			return loss
