import numpy as np

CLASS_NAMES = ['benh_nhan','cua','khong', 'nguoi', \
                'test_benh_nhan', 'test_cua','test_khong', 'test_nguoi']
TRANSMAT_PRIOR = np.array([
                [0.1,0.5,0.1,0.1,0.1,0.1,],
                [0.1,0.1,0.5,0.1,0.1,0.1,],
                [0.1,0.1,0.1,0.5,0.1,0.1,],
                [0.1,0.1,0.1,0.1,0.5,0.1,],
                [0.1,0.1,0.1,0.1,0.1,0.5,],
                [0.1,0.1,0.1,0.1,0.1,0.5,],
            ])
START_PROB = np.array([0.7,0.2,0.1,0.0,0.0,0.0])

DATA_PATH = 'hmm_data'


def get_gt(name):
    if 'benh_nhan' in name:
        return 0
    elif 'cua' in name:
        return 1
    elif 'khong' in name:
        return 2
    elif 'nguoi' in name:
        return 3