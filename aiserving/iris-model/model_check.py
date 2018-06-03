# encoding: utf-8

import requests
from sklearn import datasets
import os

TEST_SIZE = 50

cur_dir = os.path.dirname(__file__)


api_url = "http://127.0.0.1:8080/predict/"

def check_iris_model():
    all_f, all_t = datasets.load_iris(return_X_y=True)

    for f,t in zip(all_f, all_t):
        print(f)
        post_body = dict(zip(["sepal_len", "sepal_wid", "petal_len", "petal_wid"],f))
        pred_res = requests.post(api_url,json=post_body, headers={"content-type": "application/json"})
        print(pred_res.json(), t)

check_iris_model()
