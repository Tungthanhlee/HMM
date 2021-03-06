import numpy as np

CLASS_NAMES = ['benh_nhan','cua','khong', 'nguoi', \
                'test_benh_nhan', 'test_cua','test_khong', 'test_nguoi']


#benh_nhan
N_COMPONENT_BN = 6
START_PROB_BN = np.array([0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
TRANSMAT_PRIOR_BN = np.array([
            [0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6,0.2,0.1,0.1],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            ])
#cua
N_COMPONENT_CUA = 3
START_PROB_CUA = np.array([0.7,0.2,0.1,0.0,0.0,0.0, 0.0,0.0,0.0])
TRANSMAT_PRIOR_CUA = np.array([
            [0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            ])
#khong
N_COMPONENT_KHONG = 3
START_PROB_KHONG=np.array([0.5,0.4,0.1,0.0,0.0,0.0, 0.0,0.0,0.0])
TRANSMAT_PRIOR_KHONG=np.array([
            [0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            ])
#nguoi
N_COMPONENT_NGUOI = 4
START_PROB_NGUOI=np.array([0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
TRANSMAT_PRIOR_NGUOI=np.array([
            [0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.4,0.1,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            ])


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