# src/ci_processor/utils.py

import numpy as np

def detect_system_type(path):
    path = path.lower()
    if 'cochlear' in path:
        return 'Cochlear'
    elif 'ab' in path:
        return 'AB'
    else:
        return 'Unknown'

def load_dat_file(dat_file_path):
    try:
        X = np.load(dat_file_path)
    except:
        X = np.fromfile(dat_file_path, dtype=np.float32)
    return X
