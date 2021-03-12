import json
import numpy as np


# return dict
def open_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# 14 -> 14 dim one-hot vector
def make_onehot_encoding(vec, classes):
    ret = np.eye(classes)[vec]
    return ret
