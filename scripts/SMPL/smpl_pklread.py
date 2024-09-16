import pickle

import chumpy
import numpy as np

smpl_path = "load/smplx/SMPL_NEUTRAL.pkl"


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


with open(smpl_path, "rb") as smpl_file:
    data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))
