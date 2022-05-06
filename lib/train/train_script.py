import os
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.train.base_functions import *
from lib.models import build_cswintt
from lib.train.actors import CSWinTTActor,CSWinTT_T_Actor
import importlib

def run(settings):

	# update the default configs with config file
	if not os.path.exists(settings.cfg_file):
		raise ValueError("%s doesn't exist." % settings.cfg_file)
	config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
	cfg = config_module.cfg
	config_module.update_config_from_file(settings.cfg_file)
	if settings.local_rank in [-1, 0]:
		print("New configuration is shown below.")
		for key in cfg.keys():
			print("%s configuration:" % key, cfg[key])
			print('\n')

	# update settings based on cfg
	update_settings(settings, cfg)

	# Record the training log
	log_dir = os.path.join(settings.save_dir, 'logs')
	if settings.local_rank in [-1, 0]:
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
	settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

	# Build dataloaders
	loader_train, loader_val = build_dataloaders(cfg, settings)
	# Create network
	net = build_cswintt(cfg)
	net.cuda()

	# wrap networks to distributed one
	if settings.local_rank != -1:
		net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
		settings.device = torch.device("cuda:%d" % settings.local_rank)
	else:
		settings.device = torch.device("cuda:0")

	# Loss functions and Actors
	if settings.script_name == "cswintt":
		objective = {'giou': giou_loss, 'l1': l1_loss}
		loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
		actor = CSWinTTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
	elif settings.script_name == "cswintt_cls":
		objective = {'cls': BCEWithLogitsLoss()}
		loss_weight = {'cls': 1.0}
		actor = CSWinTT_T_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
	else:
		raise ValueError("illegal script name")

	if cfg.TRAIN.DEEP_SUPERVISION:
		raise ValueError("Deep supervision is not supported now.")

	# Optimizer, parameters, and learning rates
	optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

	trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

	if settings.script_name == "cswintt":
		trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)
	else:
		trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)