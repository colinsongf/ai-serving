# encoding:utf-8
import os
import hashlib

import numpy


def mkdir_if_not_exists(dir_path):
    pass


def remove_dir_if_exists(dir_path):
    pass


def copy_file_to_dir(file_path, dir_path):
    pass


def copy_dir_under_dir(src_dir, target_dir):
    pass


def get_clean_file_name(file_path, with_ext=False):
    file_name, extension = os.path.splitext(os.path.basename(file_path))
    if not with_ext:
        return file_name
    else:
        return file_name, extension


def get_save_bin_name(alog_name, input_csv, out_csv, para_dict):
    i = None
    if type(input_csv).__name__ == list.__name__:
        i = "_".join([get_clean_file_name(f) for f in input_csv])
    else:
        i = get_clean_file_name(input_csv)
    o = get_clean_file_name(out_csv)
    unique_str = ""
    for (key, val) in sorted(para_dict.items()):
        val_str = str(val)
        if type(val).__name__ == "dict":
            val_str = str(sorted(val.items()))
        unique_str += str(key) + val_str
    unique_str = "_".join([o, i, alog_name, unique_str])
    unique_str_id = hashlib.sha1(unique_str.encode("utf-8")).hexdigest()
    return "_".join([o, unique_str_id[:5]])


def df_to_json(df):
    assert len(df) == 1, "dataframe to json with more than one row"
    return {
        k: df[k][0] if not isinstance(df[k][0], numpy.generic) else numpy.asscalar(df[k][0])
        for k in df.columns
    }


# 要根据输入名字和参数名字自动生成输出名字
def generate_transformer_out_names(name, algo_nam, input_info, para_dict):
    output_file_name = None
    output_model_name = None
    pass
    return output_file_name, output_model_name


def analysis_cmd_info(cmd_str):
    pass


def get_real_cmd(algo_name_str, cmd_dict):
    pass
