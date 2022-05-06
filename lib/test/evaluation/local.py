from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    workspace_root = ''
    data_root = ''

    settings.davis_dir = ''
    settings.got10k_lmdb_path = data_root+'got10k_lmdb'
    settings.got10k_path = data_root+'got-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = data_root+'lasot_lmdb'
    settings.lasot_path = data_root+'LaSOT'
    settings.network_path = workspace_root+'output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = data_root+'OTB'
    settings.prj_dir = workspace_root
    settings.result_plot_path = workspace_root+'output/test/result_plots'
    settings.results_path = workspace_root+'output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = workspace_root+'output'
    settings.segmentation_path = workspace_root+'output/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = data_root+'TrackingNet'
    settings.uav_path = data_root + 'UAV123'
    settings.vot_path = data_root+'VOT/VOT2018'
    settings.youtubevos_dir = ''

    return settings

