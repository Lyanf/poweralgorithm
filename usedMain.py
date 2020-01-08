# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import  datetime
from hmmlearn.hmm import GaussianHMM
from matplotlib.pylab import style
from scipy.stats.stats import pearsonr
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
from Tool import Tool

def getEntropy(s):
    # 找到各个不同取值出现的次数
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = s.groupby(by=s).count().values / float(len(s))
    # prt_ary = pd.groupby(s , by = s).count().values / float(len(s))
    return -(np.log2(prt_ary) * prt_ary).sum()

def phqCluster(data):
    m, n = data.shape
    kmeans = KMeans(n_clusters=5, random_state=1).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_


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
def train_forecast(P_total, corr_device, device_index,day_point):

    # day_point = 480  # 一天为480个数据点
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
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = model_selection.train_test_split(X_norm, y_norm, test_size=0.3)

    

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


def cluster(data, day_point):
    hour_point = day_point // 24
    data_hour = data[:len(data) // hour_point * hour_point].reshape([-1, hour_point])
    data_day = data[:len(data) // day_point * day_point].reshape([-1, day_point])
    # 先进行各时序norm再聚类
    data_hour = normalize(data_hour, axis=1, norm='max')
    data_day = normalize(data_day, axis=1, norm='max')
    kmeans_hour, labels_hour = phqCluster(data_hour)
    kmeans_day, labels_day = phqCluster(data_day)
    # plt.figure(figsize=(16, 8))
    # for i in range(5):
    #     plt.plot(kmeans_hour[i], label="cluster" + str(i))
    # plt.legend()
    # plt.title('cluster_hour')
    # plt.figure(figsize=(16, 8))
    # for i in range(5):
    #     plt.plot(kmeans_day[i], label="cluster" + str(i))
    # plt.legend()
    # plt.title('cluster_day')
    return kmeans_hour, labels_hour, kmeans_day, labels_day
    # return hourList,dayList,kmeans_hour,labels_hour,kmeans_day,labels_day


def profileFeature(data, kmeans_hour, kmeans_day, labels_hour, labels_day, temp8760):
    staticFeatures = [data.max(), data.min(), data.median(), data.mean(), data.std(), np.mean(np.fft.fft(data)),
                      np.std(np.fft.fft(data)), kmeans_hour, kmeans_day]

    n_hidden_states = 5
    hmm_hour = GaussianHMM(n_components=n_hidden_states)
    hmm_hour.fit(labels_hour.reshape(-1, 1))
    transmat_hour = hmm_hour.transmat_  # 转移特性矩阵
    entropy_hour = getEntropy(labels_hour)  # 行为信息熵
    hmm_day = GaussianHMM(n_components=n_hidden_states)
    hmm_day.fit(labels_day.reshape(-1, 1))
    transmat_day = hmm_day.transmat_  # 转移特性矩阵
    entropy_day = getEntropy(labels_day)  # 行为信息熵
    dynamicFeatures = [transmat_hour, entropy_hour, transmat_day, entropy_day]

    tempload, temp, scattertemp, scatterdataunique = plotTempFeature(data, temp8760)
    return staticFeatures, dynamicFeatures, tempload, temp, scattertemp, scatterdataunique

def getData(data, date, delta, day_point):
    date -= datetime.timedelta(days=delta)
    dateStr = str(date.year) + '-' + "%0.2d"%(date.month) + '-' + "%0.2d"%(date.day)
    res = data[dateStr]
    count = 0
    while len(res) != day_point:
        date -= datetime.timedelta(days=1)
        dateStr = str(date.year) + '-' + "%0.2d"%(date.month) + '-' + "%0.2d"%(date.day)
        res = data[dateStr]
        count += 1
        if count > 10:
            return 0
    return np.array(res)


def baseline(data, year, month, day, day_point):
    date = datetime.datetime(year, month, day)
    data1, data2, data3, data7 = getData(data, date, 1, day_point), getData(data, date, 2, day_point), getData(data,
                                                                                                               date, 3,
                                                                                                               day_point), getData(
        data, date, 7, day_point)
    # print(data1,data2, data3, data7)
    res = (data1 + data2 + data3 + data7) / 4  # 该设备该日期的能耗基线

    # plt.figure(figsize=(16, 8))
    # plt.plot(res, label="能耗基线")
    # plt.plot(np.array(data[str(date.year) + '-' + str(date.month) + '-' + str(date.day)]), label="实际值")
    # plt.legend()
    # plt.title('能耗基线与实际值对比')
    return res, np.array(data[str(date.year) + '-' + str(date.month) + '-' + str(date.day)])

def plotTempFeature(data, temp8760):
    data.index = data.index.strftime("%Y-%m-%d %H")
    data_unique = data[~data.index.duplicated()]
    begin = data_unique.index[0]
    dd_begin = datetime.datetime.strptime(begin, "%Y-%m-%d %H")
    index_begin = (dd_begin.timetuple().tm_yday - 1) + dd_begin.timetuple().tm_hour
    temp = temp8760[index_begin:index_begin + data_unique.shape[0]].squeeze()
    # plt.figure(figsize=(16, 8))
    # plt.scatter(temp, data_unique)
    # plt.title('负荷温度特性')
    # plt.xlabel('温度(℃)')
    # plt.ylabel('负荷(KW)')
    #
    # fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(111)
    # lns1 = ax.plot(list(range(200)), data_unique[:200], label='负荷', color='red')
    # ax2 = plt.twinx()
    # lns2 = ax2.plot(list(range(200)), temp[:200], label='温度', color='blue')
    # lns = lns1 + lns2
    # labs = [l.get_label() for l in lns]
    # ax.legend(loc=0)
    # ax.legend(lns, labs, loc=0)
    return data_unique[:200], temp[:200], temp, data_unique

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
    P_total, device_index = Tool.getP_total(factory, line, device, measurePoint)
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
