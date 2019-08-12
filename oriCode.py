# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats.stats import pearsonr
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# 读取数据
from Tool import Tool


def pearsonLagSingle(P_total, i, j, timelag):  # 时滞-timelag~timelag间的最大pearson
    if timelag < 0:
        timelag = -timelag
    maxPearson = pearsonr(P_total.iloc[:, i], P_total.iloc[:, j])[0]
    for lag in range(1, timelag + 1):
        cand1 = pearsonr(P_total.iloc[lag:, i], P_total.iloc[:-lag, j])[0]
        cand2 = pearsonr(P_total.iloc[:-lag, i], P_total.iloc[lag:, j])[0]
        maxPearson = max(maxPearson, cand1, cand2)
    return maxPearson


# 求i设备的最相关的N个设备(以带时滞的相关系数pearsonLag为例，若需加快可直接降低timelag)
def correlation(P_total, i, N, timelag=50):
    # pearson = P_total.corr()
    # cov = P_total.cov()
    device_pearsonLag = [-1 for j in range(P_total.shape[1])]
    for j in range(P_total.shape[1]):
        if j == i:
            continue
        device_pearsonLag[j] = pearsonLagSingle(P_total, i, j, timelag)

    if N >= len(device_pearsonLag):
        N = len(device_pearsonLag) - 1
    device_pearsonLag = pd.Series(data=device_pearsonLag, index=P_total.columns)
    device_pearsonLag.sort_values(ascending=False, inplace=True)

    # plt.figure(figsize=(16, 8))
    # plt.plot(P_total.iloc[:, i], label="forecast device")
    # plt.plot(P_total.loc[:, device_pearsonLag.index[0]], label="corr device 1 : " + str(device_pearsonLag[0]))
    # plt.plot(P_total.loc[:, device_pearsonLag.index[1]], label="corr device 2 : " + str(device_pearsonLag[1]))
    # plt.plot(P_total.loc[:, device_pearsonLag.index[2]], label="corr device 3 : " + str(device_pearsonLag[2]))
    # plt.legend()
    # plt.title('corr device with correlation coefficient')
    return device_pearsonLag[:N]

# 返回的是两个list，分别为真实值和预测值
def train_forecast(P_total, corr_device, device_index):
    day_point = 480  # 一天为480个数据点
    P_forecast = P_total.iloc[:, device_index]
    y_total = P_forecast[day_point * 7:].reset_index(drop=True)
    X_total = pd.DataFrame(index=range(len(y_total)))
    timeStamp = pd.Series(P_forecast[day_point * 7:].index)
    # 生成输入特征集
    X_total['month'] = timeStamp.map(lambda x: x.month)
    X_total['weekday'] = timeStamp.map(lambda x: x.day)
    X_total['hour'] = timeStamp.map(lambda x: x.hour)
    X_total['7dAgo'] = P_forecast[:-day_point * 7].reset_index(drop=True)
    X_total['1dAgo'] = P_forecast[day_point * 6:-day_point * 1].reset_index(drop=True)
    for i in range(len(corr_device)):
        P_corr = P_total.loc[:, corr_device.index[i]]
        X_total['7dAgo_corr' + str(i)] = P_corr[:-day_point * 7].reset_index(drop=True)
        X_total['1dAgo_corr' + str(i)] = P_corr[day_point * 6:-day_point * 1].reset_index(drop=True)
    # min-max归一化
    X_norm = (X_total - X_total.min()) / (X_total.max() - X_total.min())
    y_norm = (y_total - y_total.min()) / (y_total.max() - y_total.min())
    y_min = y_total.min()
    y_max = y_total.max()
    # 分割训练集测试集
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = model_selection.train_test_split(X_norm, y_norm,
                                                                                            test_size=0.3)
    y_test = y_test_norm * (y_max - y_min) + y_min
    y_train = y_train_norm * (y_max - y_min) + y_min
    # 训练与测试
    # estimator = RandomForestRegressor(n_estimators=1000,n_jobs=-1).fit(X_train_norm, y_train_norm)
    other_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.6, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    estimator = xgb.XGBRegressor(**other_params).fit(X_train_norm, y_train_norm)
    # 测试集误差
    y_predict_test_norm = estimator.predict(X_test_norm)
    y_predict_test = y_predict_test_norm * (y_max - y_min) + y_min
    MAPE_test = np.mean(abs(y_predict_test - y_test) / y_test) * 100
    RMSE_test = np.sqrt(np.mean((y_predict_test - y_test) ** 2))
    print('MAPE_test: %f, RMSE_test: %f', MAPE_test, RMSE_test)
    # 训练集误差
    y_predict_train_norm = estimator.predict(X_train_norm)
    y_predict_train = y_predict_train_norm * (y_max - y_min) + y_min
    MAPE_train = np.mean(abs(y_predict_train - y_train) / y_train) * 100
    RMSE_train = np.sqrt(np.mean((y_predict_train - y_train) ** 2))
    print('MAPE_train: %f, RMSE_train: %f', MAPE_train, RMSE_train)

    # 预测之后一天的负荷
    # 生成之后一天的输入特征集
    X_nextday = X_total.iloc[-480:, :].reset_index(drop=True)
    X_nextday['7dAgo'] = P_forecast[-day_point * 7:-day_point * 6].reset_index(drop=True)
    X_nextday['1dAgo'] = P_forecast[-day_point:].reset_index(drop=True)
    for i in range(len(corr_device)):
        P_corr = P_total.loc[:, corr_device.index[i]]
        X_nextday['7dAgo_corr' + str(i)] = P_corr[-day_point * 7:-day_point * 6].reset_index(drop=True)
        X_nextday['1dAgo_corr' + str(i)] = P_corr[-day_point:].reset_index(drop=True)
    X_nextday_norm = (X_nextday - X_total.min()) / (X_total.max() - X_total.min())
    y_predict_nextday_norm = estimator.predict(X_nextday_norm)
    y_predict_nextday = y_predict_nextday_norm * (y_max - y_min) + y_min

    # 以最后一周(加之后一天)为例作图
    y_predict_norm = estimator.predict(X_norm)
    y_predict = y_predict_norm * (y_max - y_min) + y_min
    plt.figure(figsize=(16, 8))
    plt.plot(np.array(y_total)[-7 * day_point:], label="y_true")
    plt.plot(np.concatenate((y_predict[-7 * day_point:], y_predict_nextday), axis=0), label="y_predict")
    plt.legend()
    a, b = np.array(y_total)[-7 * day_point:], np.concatenate((y_predict[-7 * day_point:], y_predict_nextday), axis=0)
    # with open("predict.json","w") as f:
    #     json.dump({'y_true':a.tolist(),'y_pred':b.tolist()},f)
    # f.write("\n".join([a.tolist(),b.tolist()]))
    return a.tolist(), b.tolist()


def cluster(data):
    data_hour = data[:len(data) // 20 * 20].reshape([-1, 20])
    data_day = data[:len(data) // 480 * 480].reshape([-1, 480])
    # 先进行各时序norm再聚类
    data_hour = normalize(data_hour, axis=1, norm='max')
    data_day = normalize(data_day, axis=1, norm='max')
    kmeans_hour = KMeans(n_clusters=5, random_state=1).fit(data_hour)
    kmeans_day = KMeans(n_clusters=5, random_state=1).fit(data_day)
    plt.figure(figsize=(16, 8))
    hourList = []
    dayList = []
    for i in range(5):
        hourList.append(kmeans_hour.cluster_centers_[i].tolist())
        plt.plot(kmeans_hour.cluster_centers_[i], label="cluster" + str(i))
    plt.legend()
    plt.title('cluster_hour')
    plt.figure(figsize=(16, 8))
    for i in range(5):
        dayList.append(kmeans_day.cluster_centers_[i].tolist())
        plt.plot(kmeans_day.cluster_centers_[i], label="cluster" + str(i))
    plt.legend()
    plt.title('cluster_day')
    # plt.show(block = True)
    return hourList,dayList


# @click.command()
# @click.option('--name',default='三相总有功功率')
def oldMain(factory, line, device, measurePoint='三相总有功功率'):
    # print(factory)
    # print(line)
    # print(device)
    #
    # dataDir = "data/"+factory+"/"
    # # measurePoint = ''  # 所分析数据项
    # # device = '低压总出'  # 所要预测设备(全建筑总出)
    # # if os.path.exists('data\\tmp\\P_total.csv'):
    # #     P_total = pd.read_csv('data\\tmp\\P_total.csv', index_col=0)
    # # else:
    # P_total = readData(dataDir, (measurePoint))
    # P_total.to_csv('data/tmp/P_total.csv')
    # P_total.index = pd.to_datetime(P_total.index)
    # # 补全device名并得到device_index
    # for i in range(P_total.shape[1]):
    #     if device in P_total.columns[i]:
    #         device = P_total.columns[i]
    #         device_index = i
    # if device_index == -1:
    #     raise NameError('Check the device!')

    # 不需要时间参数
    # 功能一：基于自适应时滞pearson相关系数找最相关设备
    P_total,device_index = Tool.getP_total(factory, line, device, measurePoint)
    print("—————————————————一、时空相关性分析(图1)—————————————————————")
    corr_device = correlation(P_total, device_index, 3)
    print('corr_device:', corr_device)

    # 功能二：进行负荷预测模型的训练与测试
    # 需要返回什么数据/模型可自行修改函数
    print("—————————————————二、用户负荷建模与预测(图2)—————————————————————")
    a, b = train_forecast(P_total, corr_device, device_index)
    return a.tolist(), b.tolist()

    # 功能三：以小时/天尺度对负荷数据进行聚类
    # 不需要时间参数
    # 需要返回什么数据/模型可自行修改函数
    # print("—————————————————三、多时间尺度用能模式挖掘(图3图4)—————————————————————")
    # cluster(np.array(P_total.iloc[:, device_index]))

# if __name__=='__main__':
#     oldMain('常州天和印染有限公司2','','低压总出')
