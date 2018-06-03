# -*- coding: utf-8 -*-

try:
    import configparser
except:
    import ConfigParser as configparser

import json
from .model_loader import load_model


class PredictException(Exception):
    pass


'''BinModel所有上线服务模型都要继承实现的类'''


class BinModel():
    def __init__(self, model_code, model_config_path):
        '''
        加载配置文件中的必填几个配置
        :param model_code: 模型的唯一标示,英文和下划线
        :param model_config_path: 配置文件位置
        '''
        self.config = configparser.RawConfigParser()
        self.config.read(model_config_path)
        model_path = self.config.get(model_code, 'path')
        desc_path = self.config.get(model_code, "desc_path")
        self.model_bin_type = self.config.get(model_code, 'bin_type')
        self.features = self.config.get(model_code, "ordered_feature_list").split(",")
        self.desc = json.load(open(desc_path))
        self.model = self.load_model(model_path, self.model_bin_type)

    def load_model(self, model_path, bin_model_type):
        '''
        根据模型文件类型加载模型文件，当前只支持pickle和tf，其它类型的可以重写该方法
        :param model_path: 模型文件路径
        :param bin_model_type: 模型文件类型
        :return: 加载的模型
        '''
        return load_model(model_path, bin_model_type)

    def model_input_handle(self, raw_input_dict):
        '''
        将原始的输入请求转换为单条list类型的单条记录,如[1.0,2.0.3.0]
        :param raw_input_dict: 原始请求词典
        :return: 按照配置的特征顺序转换的[]
        '''
        new_input = {}
        for k in raw_input_dict.keys():
            new_input[k] = raw_input_dict[k]

        single_predict_record = []
        missed_features = []

        for f in self.features:
            if f in new_input:
                single_predict_record.append(new_input[f])
            else:
                missed_features.append(f)
        if len(missed_features) > 0:
            raise PredictException("missed input features: %s" % ",".join(missed_features))
        return single_predict_record

    def refine_model_input_handle(self, first_handled_input):
        '''对输入的原始特征加工的逻辑,输入的是单条记录，输入的是要进行模型预测的数据'''
        return first_handled_input

    def raw_predict(self, refined_input):
        '''调用模型实际预测的方法'''
        return self.model.predict_proba([refined_input])[0][1]

    def model_output_handle(self, predict_score):
        '''
        对模型的输出进行加工，返回词典类型的输出结果,key是自定义的模型输出名字，value是输出值
        '''
        return predict_score

    def model_predict(self, single_record):
        '''
        模型预测的调用流程
        :param single_record: 原始的http请求
        :return: 模型预测输出
        '''
        predict_input = self.model_input_handle(single_record)
        refined_input = self.refine_model_input_handle(predict_input)
        predict_result = self.raw_predict(refined_input)
        return self.model_output_handle(predict_result)
