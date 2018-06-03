# -*- coding: utf-8 -*-

from .model_io import get_session,load_pickle_model,load_tf_model,LoadModelException
from .logger import LOGGER

SESS_DICT = {}


BIN_TYPE = {'pickle': load_pickle_model, 'tf': load_tf_model}


def load_model(model_path, bin_model_type):
    if (bin_model_type in BIN_TYPE):
        LOGGER.info("Begin to load %s model at : %s !" % (bin_model_type, model_path))
        return BIN_TYPE[bin_model_type](model_path)
    else:
        err_msg = bin_model_type + " bin model type not support yet!"
        LOGGER.error(err_msg)
        raise LoadModelException(err_msg)
