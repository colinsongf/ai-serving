# -*- coding: utf-8 -*-

import pickle
from .logger import LOGGER
from .utils import get_clean_file_name
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
from tensorflow.python.saved_model import utils
from tensorflow.contrib.layers import create_feature_spec_for_parsing
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


SESS_DICT = {}


def get_session(model_id):
    global SESS_DICT
    config = tf.ConfigProto(allow_soft_placement=True)
    SESS_DICT[model_id] = tf.Session(config=config)
    return SESS_DICT[model_id]


class LoadModelException(Exception):
    pass


def load_pickle_model(model_path):
    return pickle.load(open(model_path, 'rb'))


def load_joblib_model(model_path):
    pass
    # return joblib.load(model_path)


def load_tf_model(model_path):
    sess = get_session(model_path)
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    return sess


def load_tf_learn_model(model_path):
    m = load_pickle_model(model_path)
    # if not os.path.exists(m.model_dir):
    #     raise LoadModelException("tf learn model checkpoint dir didn't exists!")
    return m


def load_xgb_model(model_path):
    load_pickle_model(model_path)



class bin_type():
    PICKLE = 'pickle',
    JOBLIB = 'joblib',
    TENSORFLOW = 'tf',
    TF_LEARN = 'tflearn'


BIN_TYPE_FUC_DICT = {bin_type.PICKLE: load_pickle_model, bin_type.JOBLIB: load_joblib_model,
                     bin_type.TENSORFLOW: load_tf_model, bin_type.TF_LEARN: load_tf_learn_model}


def load_model(model_path, bin_model_type):
    if bin_model_type == "pickle": bin_model_type = bin_type.PICKLE
    if bin_model_type == "joblib": bin_model_type = bin_type.JOBLIB
    if bin_model_type == "tf": bin_model_type = bin_type.TENSORFLOW
    if bin_model_type == "tflearn": bin_model_type = bin_type.TF_LEARN

    if (bin_model_type in BIN_TYPE_FUC_DICT):
        LOGGER.info("Begin to load %s model at : %s !" % (bin_model_type, model_path))
        return BIN_TYPE_FUC_DICT[bin_model_type](model_path)
    else:
        err_msg = bin_model_type + " bin model type not support yet!"
        LOGGER.error(err_msg)
        raise LoadModelException(err_msg)


def save_pickle_model(model, model_name, save_dir):
    full_save_path = os.path.join(save_dir, model_name, )
    pickle.dump(model, open(full_save_path, 'wb'))
    return full_save_path


def save_tf_model(sess, model_name, model_version, input_tensor_dict, out_tensor_dict, force_write=False):
    if (type(model_version) is not int):
        print("Error! input model_version must be a int number! eg. 1")
        return

    export_path = os.path.join(compat.as_bytes(model_name), compat.as_bytes(str(model_version)))
    if (force_write and os.path.exists(export_path)):
        shutil.rmtree(export_path)

    print('Exporting trained model to', export_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature_inputs = {key: utils.build_tensor_info(tensor)
                        for key, tensor in input_tensor_dict.items()}
    signature_outputs = {key: utils.build_tensor_info(tensor)
                         for key, tensor in out_tensor_dict.items()}

    prediction_signature = signature_def_utils.build_signature_def(
        inputs=signature_inputs,
        outputs=signature_outputs,
        method_name=signature_constants.PREDICT_METHOD_NAME)

    # legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map={model_name: prediction_signature},
                                         # legacy_init_op=legacy_init_op,
                                         clear_devices=True)
    builder.save()
    print('Done exporting!')


def save_tf_learn_model(estimator, model_name, export_dir, feature_columns, ):
    feature_spec = create_feature_spec_for_parsing(feature_columns)
    serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
    export_dir = os.path.join(export_dir, model_name)
    estimator.export_savedmodel(export_dir, serving_input_fn)
    print("Done exporting tf.learn model to " + export_dir + "!")


def infer_bin_type(model_path):
    if os.path.isdir(model_path):
        return bin_type.TENSORFLOW
    file_name, ext = get_clean_file_name(model_path, True)
    if ext == '.pkl':
        return bin_type.PICKLE
    else:
        raise Exception("can't detect model bin type at %s!" % model_path)