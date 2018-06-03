# encoding:utf-8
from wai.logger import LOGGER
from wai import utils
from wai import model_io

import os
import numpy as np

import xgboost as xgb
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import preprocessing
from sklearn import linear_model, ensemble
from sklearn import metrics
from scipy import stats
import tensorflow as tf

learn = tf.contrib.learn

CHECKPOINT_DIR = "saved/ckpt"


class ExecuteError(Exception): pass


SKLEARN_ALOG_DICT = {
    "lr": linear_model.LogisticRegression,
    "rf": ensemble.RandomForestClassifier
}


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, col_idxes=None, drop=False):
        self.act_cols_idx = col_idxes
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.act_cols_idx:
            if self.drop:
                np.delete(X, self.act_cols_idx, axis=1)
                return X
            else:
                return X[:, self.act_cols_idx]
        else:
            return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


# 定义可用的转换算法
TRASFORMER_ALOG_DICT = {
    "min_max": preprocessing.MinMaxScaler,
    "select": SelectColumns
}


# http://scikit-learn.org/stable/
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
def get_fitted_transformer(algo_name, para_dict, X, y=None, alog_dict=TRASFORMER_ALOG_DICT):
    if algo_name in alog_dict:
        return alog_dict[algo_name](**para_dict).fit(X, y)
    else:
        msg = "algo %s not supported yet!" % algo_name
        LOGGER.error(msg)
        raise ExecuteError(msg)


TRANSFORM = "transform"


def exec_transformer(trans, X):
    Xt = trans.transform(X)
    return Xt


MERGE_COLUMNS = "merge"


class MergeColumns():
    def __init__(self, input_1_cols, input_2_cols):
        self.cols_1 = input_1_cols
        self.cols_2 = input_2_cols

    def merge_list_or_1d(self, input_1, input_2):
        if isinstance(input_1, list):
            input_1 = np.array(input_1)
        if isinstance(input_2, list):
            input_2 = np.array(input_2)

        d1 = input_1.shape[0]
        d2 = input_2.shape[0]
        if d1 == 1 and d1 == d2:
            return True
        else:
            return False

    def merge(self, input_1, input_2):
        if self.merge_list_or_1d(input_1, input_2):
            return np.hstack(([input_1[i] for i in self.cols_1], [input_2[i] for i in self.cols_2]))
        return np.hstack((input_1[:, self.cols_1], input_2[:, self.cols_2]))


def get_cols_merger(para_dict):
    input_1_cols = para_dict["input_1_cols"]
    input_2_cols = para_dict["input_2_cols"]
    return MergeColumns(input_1_cols, input_2_cols)


def exec_merger(merger, input_1, input_2):
    return merger.merge(input_1, input_2)


def get_fitted_sklearn_model(algo_name, para_dict, X, y):
    return get_fitted_transformer(algo_name, para_dict, X, y, SKLEARN_ALOG_DICT)


def exec_sklearn_model(model, X):
    if hasattr(model, 'predict'):
        return model.predict(X)
    elif hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        err_msg = "sklearn model %s should have predict or predickt_proba method!" % (str(model))
        LOGGER.error(err_msg)
        raise ExecuteError(err_msg)


# http://xgboost.readthedocs.io/en/latest/python/python_intro.html#setting-parameters
def get_fitted_xgb_model(algo_name, para_dict, X, y):
    train_matrix = xgb.DMatrix(X, y)
    para_dict["dtrain"] = train_matrix
    return xgb.train(**para_dict)


def exec_xgb_model(model, X):
    xmatrix = xgb.DMatrix(X)
    return model.predict(xmatrix)


# https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
def get_fitted_tfdnn_model(model_name, para_dict, X, y):
    init_para_dict = {}
    init_para_dict["model_dir"] = os.path.join(CHECKPOINT_DIR, model_name)
    init_para_dict["hidden_units"] = para_dict["hidden_units"]
    init_para_dict["feature_columns"] = para_dict.get("feature_columns", learn.infer_real_valued_columns_from_input(X))

    fit_para_dict = {}
    fit_para_dict["steps"] = para_dict.get("steps", 200)
    dnn_clf = learn.DNNClassifier(**init_para_dict)
    dnn_clf2 = learn.DNNClassifier(**init_para_dict)
    return dnn_clf.fit(X, y, **fit_para_dict), dnn_clf2


def exec_tfdnn_model(model, X):
    return list(model.predict(X))


# 定义模型对应的训练函数
FIT_MODEL_DICT = {
    "lr": get_fitted_sklearn_model,
    "rf": get_fitted_sklearn_model,
    "xgb": get_fitted_xgb_model,
    "tfdnn": get_fitted_tfdnn_model
}

# 定义模型预测时对应的函数
PREDICT_ALOG_NAME = 'predict'
PREDICT_DICT = {
    "lr": exec_sklearn_model,
    "rf": exec_sklearn_model,
    "xgb": exec_xgb_model,
    "tfdnn": exec_tfdnn_model
}


def model_predict(input_x, para_dict):
    model_name = para_dict["model_alog_name"]
    model_path = para_dict["model_path"]
    md = model_io.load_model(model_path, utils.infer_bin_type(model_path))
    predict_y = PREDICT_DICT[model_name](md, input_x)
    return predict_y


#########################
# model evaluation part
#########################

EVALUATION_ALOG_NAME = "eval_model"
EVALUATION_FUNC_DICT = {
    "auc": metrics.auc,
    "confusion_matrix": metrics.confusion_matrix,
    "f1_score": metrics.f1_score,
    "ks": stats.ks_2samp,
    "precision": metrics.precision_score,
    "recall": metrics.recall_score
}


def eval_classifier_model(y_true, y_pred, metric_list="auc,ks,f1_score,confusion_matrix", pos_label=1):
    eval_res_dict = {}
    y_pred = (y_pred > 0.5).astype(np.int)
    for m in metric_list.split(","):
        if m not in EVALUATION_FUNC_DICT:
            err_msg = "evaluation metric %s not supported yet!" % m
            LOGGER.error(err_msg)
            raise ExecuteError(err_msg)
        if m == "auc":
            import operator
            pred_true = np.array(sorted(list(zip(y_pred, y_true)), key=operator.itemgetter(1)))
            y_true, y_pred = pred_true[:, 1], pred_true[:, 0]

        if m == "confusion_matrix":
            import pandas as pd
            eval_res_dict[m] = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        else:
            eval_res_dict[m] = EVALUATION_FUNC_DICT[m](y_true, y_pred)
    return eval_res_dict


#########################
# test part
#########################

def _test():
    from model_io import load_model, save_pickle_model, save_tf_learn_model
    import model_io
    def get_here_path(relative_path):
        cur_dir_path = os.path.dirname(__file__)
        return os.path.join(cur_dir_path, relative_path)

    save_dir = get_here_path("saved")

    pkl_bin_type = "pickle"
    x, y = np.random.randn(20, 3), np.random.randint(2, size=20)

    # test trans
    trans = get_fitted_transformer("min_max", {"feature_range": (-1, 1)}, x)
    fitted_x = exec_transformer(trans, x)
    print(fitted_x)
    save_pickle_model(trans, "min_max_1.pkl", save_dir)

    trans = load_model(get_here_path("saved/min_max_1.pkl"), model_io.bin_type.PICKLE)
    print(exec_transformer(trans, x))

    # test sklearn model
    lr = get_fitted_sklearn_model("lr", {"penalty": 'l1'}, x, y)
    save_pickle_model(lr, "lr_1.pkl", save_dir)

    lr = load_model(get_here_path("saved/lr_1.pkl"), model_io.bin_type.PICKLE)
    print(lr)
    print(exec_sklearn_model(lr, x))

    rf = get_fitted_sklearn_model("rf", {}, x, y)
    print(rf)
    print(exec_sklearn_model(rf, x))

    # test xgb model
    para_dict = {}
    para_dict["params"] = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    para_dict["num_boost_round"] = 2

    xgb = get_fitted_xgb_model("xgb_1", para_dict, x, y)
    save_pickle_model(xgb, "xgb_1.pkl", save_dir)

    xgb = load_model(get_here_path("saved/xgb_1.pkl"), model_io.bin_type.PICKLE)
    print(str(xgb))
    print(exec_xgb_model(xgb, x))

    # test tf dnn classifier
    para_dict = {}
    para_dict["hidden_units"] = [10, 20, 10]
    para_dict["n_classes"] = 2
    dnn, dnn2 = get_fitted_tfdnn_model("dnn_1", para_dict, x.astype(np.float32), y.astype(np.float32))
    save_pickle_model(dnn2, "dnn_1.pkl", save_dir)
    save_tf_learn_model(dnn, "dnn_1", save_dir, learn.infer_real_valued_columns_from_input(x.astype(np.float32)), )
    print(list(dnn.predict(x.astype(np.float32), as_iterable=False)))

    print("begin to load dnn...")
    m = model_io.load_model("saved/dnn_1.pkl", model_io.bin_type.TF_LEARN)
    print(list(m.predict(x.astype(np.float32))))


def _test_dnn():
    import model_io
    m = model_io.load_model("saved/dnn_1.pkl", model_io.bin_type.TF_LEARN)
    x, y = np.random.randn(20, 3), np.random.randint(2, size=20)
    print(m.get_params())
    print(m.get_variables())
    print(list(m.predict(x.astype(np.float32))))


def _test_eval():
    y_pred = np.array([0, 1, 1])
    y_true = np.array([1, 0, 1])
    labels = [1, 0]
    print(eval_classifier_model(y_true, y_pred))
    # print(metrics.confusion_matrix(y_true, y_pred, labels))

    # _test()
    # _test_dnn()
    # _test_eval()

