import pandas as pd
import numpy as np
import xgboost as xgb
from math import log as ln
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from pylab import mpl


# 构造数据集
def Data():
    data = pd.read_csv('data_numeric.csv', header=None)
    x_data = data.iloc[:, 0:24]
    y_data = data.iloc[:, 24]
    model_smote = SMOTE()
    x_smote, y_smote = model_smote.fit_sample(x_data, y_data)
    x_smote = pd.DataFrame(x_smote)
    y_smote = pd.Series(y_smote)
    x_train, x_test, y_train, y_test = train_test_split(
        x_smote, y_smote, test_size=0.2)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = Data()


def param_test1(x, y):
    param_test = {'learning_rate': np.arange(0.01, 0.4, 0.02)}
    gsearch = GridSearchCV(
        estimator=xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            seed=27),
        param_grid=param_test,
        scoring='roc_auc',
        cv=5)
    gsearch.fit(x, y)
    return gsearch.best_params_, round(gsearch.best_score_, 3)


param_test1 = param_test1(x_train, y_train)


def param_test2(x, y):
    param_test = {'n_estimators': range(20, 200, 10)}
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
        learning_rate=round(param_test1[0]['learning_rate'], 2),
        max_depth=5, min_child_weight=1,
        gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', seed=27),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch.fit(x, y)
    return gsearch.best_params_, round(gsearch.best_score_, 3)


param_test2 = param_test2(x_train, y_train)


def param_test3(x, y):
    param_test = {
        'max_depth': range(
            3, 14, 1), 'min_child_weight': range(
            1, 6, 1)}
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
        learning_rate=round(param_test1[0]['learning_rate'], 2),
        n_estimators=param_test2[0]['n_estimators'],
        gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', seed=27),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch.fit(x, y)
    return gsearch.best_params_, round(gsearch.best_score_, 3)


param_test3 = param_test3(x_train, y_train)


def param_test4(x, y):
    param_test = {'gamma': np.arange(0, 0.7, 0.1)}
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
        learning_rate=round(param_test1[0]['learning_rate'], 2),
        n_estimators=param_test2[0]['n_estimators'],
        max_depth=param_test3[0]['max_depth'],
        min_child_weight=param_test3[0]['min_child_weight'],
        subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', seed=27),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch.fit(x, y)
    return gsearch.best_params_, round(gsearch.best_score_, 3)


param_test4 = param_test4(x_train, y_train)


def param_test5(x, y):
    param_test = {
        'subsample': np.arange(
            0.6, 0.9, 0.1), 'colsample_bytree': np.arange(
            0.6, 0.9, 0.1)}
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
        learning_rate=round(param_test1[0]['learning_rate'], 2),
        n_estimators=param_test2[0]['n_estimators'],
        max_depth=param_test3[0]['max_depth'],
        min_child_weight=param_test3[0]['min_child_weight'],
        gamma=round(param_test4[0]['gamma'], 1),
        objective='binary:logistic', seed=27),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch.fit(x, y)
    return gsearch.best_params_, round(gsearch.best_score_, 3)


param_test5 = param_test5(x_train, y_train)


def param_test6(x, y):
    param_test = {
        'reg_alpha': [
            0.01,
            0.1,
            1,
            10],
        'reg_lambda': range(
            1,
            10,
            2)}
    gsearch = GridSearchCV(estimator=xgb.XGBClassifier(
        learning_rate=round(param_test1[0]['learning_rate'], 2),
        n_estimators=param_test2[0]['n_estimators'],
        max_depth=param_test3[0]['max_depth'],
        min_child_weight=param_test3[0]['min_child_weight'],
        gamma=round(param_test4[0]['gamma'], 1),
        subsample=round(param_test5[0]['subsample'], 1),
        colsample_bytree=round(param_test5[0]['colsample_bytree'], 1),
        objective='binary:logistic', seed=27),
        param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch.fit(x, y)
    return gsearch.best_params_, round(gsearch.best_score_, 3)


param_test6 = param_test6(x_train, y_train)


def Model(x_train, x_test, y_train, y_test, k1, k2, k3):
    model = xgb.XGBClassifier(
        learning_rate=round(
            param_test1[0]['learning_rate'],
            2),
        n_estimators=param_test2[0]['n_estimators'],
        max_depth=param_test3[0]['max_depth'],
        min_child_weight=param_test3[0]['min_child_weight'],
        gamma=round(
            param_test4[0]['gamma'],
            1),
        subsample=round(
            param_test5[0]['subsample'],
            1),
        colsample_bytree=round(
            param_test5[0]['colsample_bytree'],
            1),
        reg_alpha=param_test6[0]['reg_alpha'],
        reg_lambda=param_test6[0]['reg_lambda'],
        objective='binary:logistic',
        seed=27)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    auc_score = round(metrics.roc_auc_score(y_test, y_pred_prob[:, 1]), 3)
    cm = metrics.confusion_matrix(y_test, y_pred)
    imb_error1 = (cm[0, 1] + cm[1, 0] * k1) / y_test.count()
    imb_error2 = (cm[0, 1] + cm[1, 0] * k2) / y_test.count()
    imb_error3 = (cm[0, 1] + cm[1, 0] * k3) / y_test.count()
    return auc_score, imb_error1, imb_error2, imb_error3, cm, y_pred_prob


xgb_model = Model(x_train, x_test, y_train, y_test, 3, 6, 10)


def Result1():
    score_result = pd.DataFrame({'AUC值': [xgb_model[0]], '代价敏感错误率平均变动幅度': [round(
        ((xgb_model[2] - xgb_model[1]) / 3 + (xgb_model[3] - xgb_model[2]) / 4) / 2, 3)]})
    param_score = {'最优参数': [param_test1[0]['learning_rate'],
                            param_test2[0]['n_estimators'],
                            param_test3[0]['max_depth'],
                            param_test3[0]['min_child_weight'],
                            param_test4[0]['gamma'],
                            param_test5[0]['subsample'],
                            param_test5[0]['colsample_bytree'],
                            param_test6[0]['reg_alpha'],
                            param_test6[0]['reg_lambda']],
                   '参数得分': [param_test1[1], param_test2[1], param_test3[1],
                            param_test3[1], param_test4[1], param_test5[1],
                            param_test5[1], param_test6[1], param_test6[1]]}
    param_score = pd.DataFrame(
        param_score,
        index=[
            'learning_rate',
            'n_estimators',
            'max_depth',
            'min_child_weight',
            'gamma',
            'subsample',
            'colsample_bytree',
            'reg_alpha',
            'reg_lambda'])
    cm = xgb_model[4]
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('预测结果的混淆矩阵', y=1.1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm.T[i, j], xy=(
                i, j), horizontalalignment='center', verticalalignment='center')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.show()
    return score_result, param_score


result1 = Result1()
print(result1[0])
print(result1[1])


def Result2(prob, theta, p, delta):
    prob_ratio = prob[:, 1] / (1 - prob[:, 1])
    b = delta / ln(2)
    a = p + b * ln(theta)
    credit_score = []
    for i in range(len(prob_ratio)):
        sc = a - b * ln(prob_ratio[i])
        credit_score.append(int(sc))
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    credit_score = np.array(credit_score)
    min_value = credit_score.min()
    max_value = credit_score.max()
    mean_value = round(credit_score.mean(), 2)
    bad_ratio = round(
        len(credit_score[credit_score < 60]) / len(credit_score), 3) * 100
    score = pd.DataFrame({'最小值': [min_value], '最大值': [max_value], '均值': [
                         mean_value], '风险客户比率': ['{}%'.format(bad_ratio)]})
    descri = pd.cut(credit_score, bins, right=False)
    des = descri.value_counts()
    return score, des


result2 = Result2(xgb_model[5], 0.7, 60, 5)
print(result2[0])
print(result2[1])
