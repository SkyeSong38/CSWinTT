# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : load_pretrain.py
# Copyright (c) Skye-Song. All Rights Reserved
import os
import torch
def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters
	share common prefix 'module.' '''
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def _get_prefix_dic(dict, prefix):
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in dict.items() if key.startswith(prefix)}


def _add_prefix_dic(dict, prefix):
	f = lambda x: str(prefix) + str(x)
	return {f(key): value for key, value in dict.items()}

def load_my_tracker(net):
	checkpoint_path = '/home/szk/Developer/trans_tracking/Stark-main/output/checkpoints/TRACKER_MY_ep0120.pth.tar'
	checkpoint_path_raw = '/home/szk/Developer/trans_tracking/Stark-main/output/checkpoints/STARKST_ep0500.pth.tar'

	check_old = torch.load(checkpoint_path, map_location='cpu')['net']
	check_decorder = torch.load(checkpoint_path_raw, map_location='cpu')['net']
	net.backbone.load_state_dict(_get_prefix_dic(check_old, "backbone."), strict=True)
	net.bottleneck.load_state_dict(_get_prefix_dic(check_old, "bottleneck."), strict=True)
	net.transformer.transformer.load_state_dict(_get_prefix_dic(check_old, "transformer.transformer."), strict=True)
	net.transformer.transformer_post.load_state_dict(_get_prefix_dic(check_decorder, "transformer.decoder.layers."),
	                                                 strict=True)
	net.transformer.post_norm.load_state_dict(_get_prefix_dic(check_decorder, "transformer.decoder.norm."),
	                                          strict=True)
	net.box_head.load_state_dict(_get_prefix_dic(check_decorder, "box_head."), strict=True)
	net.cls_head.load_state_dict(_get_prefix_dic(check_decorder, "cls_head."), strict=True)
	net.query_embed.load_state_dict(_get_prefix_dic(check_decorder, "query_embed."), strict=True)
	return net


def load_stark_for_cstrt(net):
	checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))+ '/output/checkpoints/STARKST_ep0500.pth.tar'

	check_point= torch.load(checkpoint_path, map_location='cpu')['net']
	net.backbone.load_state_dict(_get_prefix_dic(check_point, "backbone."), strict=True)
	net.bottleneck.load_state_dict(_get_prefix_dic(check_point, "bottleneck."), strict=True)
	net.box_head.load_state_dict(_get_prefix_dic(check_point, "box_head."), strict=True)
	net.cls_head.load_state_dict(_get_prefix_dic(check_point, "cls_head."), strict=True)
	net.query_embed.load_state_dict(_get_prefix_dic(check_point, "query_embed."), strict=True)

	for i in range(6):
		net.transformer.transformer[i].qkv_embedding.weight.data = check_point["transformer.encoder.layers.%s.self_attn.in_proj_weight" % (str(i))]
		net.transformer.transformer[i].qkv_embedding.bias.data = check_point["transformer.encoder.layers.%s.self_attn.in_proj_bias" % (str(i))]
		net.transformer.transformer[i].output_linear.load_state_dict(_get_prefix_dic(check_point, ("transformer.encoder.layers.%s.self_attn.out_proj." % (str(i)))), strict=True)
		net.transformer.transformer[i].feedforward[0].load_state_dict(_get_prefix_dic(check_point, ("transformer.encoder.layers.%s.linear1." % (str(i)))), strict=True)
		net.transformer.transformer[i].feedforward[3].load_state_dict(_get_prefix_dic(check_point, ("transformer.encoder.layers.%s.linear2." % (str(i)))), strict=True)
		net.transformer.transformer[i].norm1.load_state_dict(_get_prefix_dic(check_point, ("transformer.encoder.layers.%s.norm1." % (str(i)))), strict=True)
		net.transformer.transformer[i].norm2.load_state_dict(_get_prefix_dic(check_point, ("transformer.encoder.layers.%s.norm2." % (str(i)))), strict=True)

	net.transformer.transformer_post.load_state_dict(_get_prefix_dic(check_point, "transformer.decoder.layers."),
	                                                 strict=True)
	net.transformer.post_norm.load_state_dict(_get_prefix_dic(check_point, "transformer.decoder.norm."),
	                                          strict=True)

	return net
