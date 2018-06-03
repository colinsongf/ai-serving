# -*- coding: utf-8 -*-

from aiserving.bin_model import BinModel
from aiserving import serving


class IrisModel(BinModel):
    def __init__(self, model_code, model_config_path):
        super().__init__( model_code, model_config_path)

model = IrisModel("iris_model", "./config.ini")
serving.load_model("iris_model", model)
serving.run()