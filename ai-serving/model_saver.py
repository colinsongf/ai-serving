# -*- coding: utf-8 -*-

import os
import shutil
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
from tensorflow.python.saved_model import utils
from tensorflow.contrib.layers import create_feature_spec_for_parsing
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


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
        inputs = signature_inputs,
        outputs = signature_outputs,
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map={model_name: prediction_signature},
                                         clear_devices=True)
    builder.save()
    print('Done exporting!')

def save_tf_learn_model(estimator, feature_columns, export_dir):
    feature_spec = create_feature_spec_for_parsing(feature_columns)
    serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
    estimator.export_savedmodel(export_dir, serving_input_fn)
    print("Done exporting tf.learn model to " + export_dir + "!")
