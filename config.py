
conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "2",
    "data": {
        'dataset_path': "/home/huangyj_1/triplet/GaitSet-master/GaitDatasetB-silh",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 61,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 16),
        #'batch_size': (4, 8),
        'restore_iter': 0,
        'total_iter': 60000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        #'frame_num_3d': 16,
        'frame_num_3d': 30,
        'num_classes': 124,
        #'model_name': 'M3D_RAL_cat_sfxloss_ctrloss_eras_2streams',
        #'model_name': 'M3D_RAL_cat_2sfxloss_2ctrloss_temporal_25frame'
        'model_name': 'M3D_temporal'
    }
}
