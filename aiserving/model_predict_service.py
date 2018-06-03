# -*- coding: utf-8 -*-

import json
import traceback

from flask import Flask, request
from flask.json import jsonify
from .model_manager import *

app = Flask(__name__)
app.url_map.strict_slashes = False

def get_err_return(err_msg):
    LOGGER.error(err_msg)
    return json.dumps({
        "code": 1,
        "error": err_msg,
        "data": None
    })


def get_normal_return(data):
    return json.dumps({
        "code": 0,
        "error": None,
        "data": data
    })


@app.route("/check_backend_active.html")
def check_alive():
    LOGGER.info("server stat check.")
    return "OK"


@app.route("/health", methods=['GET'])
def health():
    return jsonify({"status": "up"})


@app.route("/", methods=['GET', 'POST'])
@app.route("/predict/", methods=['GET', 'POST'])
@app.route("/predict/<model_code>", methods=['GET', 'POST'])
def model_predict(model_code=None):
    if model_code == None and len(LOADED_MODELS) == 1:
        model_code = list(LOADED_MODELS.keys())[0]
    if request.method == 'GET':
        if model_code in LOADED_MODELS:
            LOGGER.info("GET method: will return: %s" % LOADED_MODELS[model_code].desc)
            return json.dumps(LOADED_MODELS[model_code].desc)
        else:
            return get_err_return("can't find the target model -> %s" % model_code)

    if request.method == 'POST':
        input_data = request.get_json(force=True, silent=False, cache=True)
        try:
            res = get_normal_return(do_model_predict(model_code, input_data))
            LOGGER.debug("input: %s\n predict result: %s" % (
                json.dumps(input_data), res))
            return res
        except PredictException as pe:
            return get_err_return("Internal model %s predict exception :%s" % (model_code, str(pe)))
    return get_err_return("not supported request method!")


@app.route("/meta", methods=['GET'])
def meta():
    model_code = None
    if len(LOADED_MODELS) == 1:
        model_code = LOADED_MODELS.keys()[0]
    if model_code in LOADED_MODELS:
        LOGGER.info("GET method: will return: %s" % LOADED_MODELS[model_code].desc)
        return json.dumps(LOADED_MODELS[model_code].desc)
    else:
        return get_err_return("can't find the target model -> %s" % model_code)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
