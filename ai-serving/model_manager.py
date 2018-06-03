# -*- coding: utf-8 -*-

try:
    import configparser
except:
    import ConfigParser as configparser

import os
import subprocess
from subprocess import CalledProcessError

from .logger import LOGGER

LOADED_MODELS = {}


class PredictException(Exception): pass


class DeployException(Exception): pass


def model_code_not_none(model_code):
    if model_code is None:
        errMsg = "None vaild model code found in url path!"
        LOGGER.error(errMsg)
        raise PredictException(errMsg)
    return True


# copy raw model data from wai to dst
# src_dir model_name/version_number/model_files
# dst_dir model_name/model_files
def deploy_model(model_code, deploy_config_path):
    config = configparser.RawConfigParser()
    config.read(deploy_config_path)
    valid_codes = config.sections()
    if model_code not in valid_codes:
        err_msg = "%s deploy failed! Can't find valid model setting in deploy_model.cfg !" % model_code
        LOGGER.error(err_msg)
        raise DeployException(err_msg)
    src_dir = config.get(model_code, "src_path")
    dst_dir = config.get(model_code, "dst_path")
    try:
        src_versions = [int(i) for i in os.listdir(src_dir)]
        version = config.get(model_code, 'src_version')
        if version == 'latest':
            version_numb = max(src_versions)
        else:
            version_numb = int(version)
        src_path = ''
        if version_numb in src_versions:
            src_path = os.path.join(src_dir, str(version_numb))
        if not os.path.exists(src_path):
            msg = " path %s didn't exist!" % src_path
            LOGGER.error(msg)
            raise DeployException(msg)
        if not os.path.exists(dst_dir):
            LOGGER.info('Target dir %s not exists, create it now!' % dst_dir)
            os.makedirs(dst_dir)
        src_path += "/*"
        dst_dir += '/'
        LOGGER.info("Ready to deploy model %s from  wai: %s to dst: %s " % (model_code, src_path, dst_dir))
        subprocess.check_call(" ".join(['cp', '-rf', src_path, dst_dir]), shell=True)
        LOGGER.info("model %s deploy finished!" % model_code)
    except CalledProcessError as cpe:
        raise DeployException(cpe)


def do_model_predict(model_code, raw_req_dict):
    try:
        if model_code not in LOADED_MODELS:
            errMsg = "model predict failed, invalid model code -> %s!" % (model_code)
            LOGGER.error(errMsg)
            raise PredictException(errMsg)
        return LOADED_MODELS[model_code].model_predict(raw_req_dict)
    except Exception as e:
        import traceback
        errMsg = "model predict failed -> %s, detail error info -> %s!" % (model_code, e)
        LOGGER.error(errMsg)
        LOGGER.exception(traceback.format_exc())
        raise PredictException(e)
