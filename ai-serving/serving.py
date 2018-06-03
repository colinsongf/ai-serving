# encoding: utf-8

import tarfile
import time


import pandas as pd
import requests
from .logger import LOGGER
from .model_predict_service import app
from .model_manager import LOADED_MODELS

def load_model(model_code, model):
    global LOADED_MODELS
    LOADED_MODELS[model_code] = model


def local_test(model_code, test_input_csv):
    api_url = "http://127.0.0.1:8080/predict/%s" % model_code
    df = pd.read_csv(test_input_csv)
    headers = list(df.columns.values)
    cost_sum = 0
    ct = 0
    for row in df.values:
        post_content = dict(zip(headers[:-1], row[:-1]))
        start = time.time()
        pred_res = requests.post(api_url, json=post_content, headers={"content-type": "application/json"})
        LOGGER.info("predict result -> ", pred_res, "actual result -> ", row[-1])
        end = time.time()
        cost_sum += (end - start) * 1000
        ct += 1
    LOGGER.info("%d test records cost total %d ms, avg cost %d ms!" % (ct, cost_sum, cost_sum / ct))


def deploy_test(model_code, root_dir):
    tar_name = model_code + ""
    tar = tarfile.open("%s.tar.gz" % tar_name, "w:gz")


def run(port=8080):
    app.run(host="0.0.0.0", port=port, threaded=True)
