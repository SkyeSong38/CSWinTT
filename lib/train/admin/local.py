class EnvironmentSettings:
    def __init__(self):
        workspace_root = ''
        data_root = ''

        self.workspace_dir = workspace_root + 'output'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = workspace_root + 'output/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = workspace_root + 'output/pretrained_networks'
        self.lasot_dir = data_root + 'LaSOT'
        self.got10k_dir = data_root + 'got-10k/train_data'
        self.lasot_lmdb_dir = data_root + 'lasot_lmdb'
        self.got10k_lmdb_dir = data_root + 'got10k_lmdb'
        self.trackingnet_dir = data_root + 'TrackingNet'
        self.trackingnet_lmdb_dir = data_root + 'trackingnet_lmdb'
        self.coco_dir = data_root + 'coco'
        self.coco_lmdb_dir = data_root + 'coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = data_root + 'Tracking/vid'
        self.imagenet_lmdb_dir = data_root + 'vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
