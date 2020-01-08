from functools import reduce
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import os, datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import style
from Tool import Tool
import time
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 上海科委项目
# 用户/建筑维度在文件
# 总维度：空间(任意指定层级) [level1、level2、...]、
#       时间[年year、月month、日day、小时hour、分钟min]、属性维metric(功率电流电压等)


def readData(dataDir):
    fileList = os.listdir(dataDir)
    res = pd.DataFrame()
    for file in fileList:
        fileDir = dataDir + file
        data = pd.read_excel(fileDir, header=1,index_col=0)
        data.index = pd.to_datetime(data.index)
        data['date'] = [datetime.datetime.strftime(x, '%Y-%m-%d') for x in data.index]
        data['month'] = [x.month for x in data.index]
        data['device'] = file
        data['user'] = "常州天和印染有限公司"
        res = res.append(data)
    res['time'] = res.index
    metricList = list(res.columns)
    metricList.remove('user')
    metricList.remove('month')
    metricList.remove('date')
    metricList.remove('device')
    metricList.remove('time')
    return res, fileList, metricList


# 切片和切块
# groups可为'day'/'metric'; agg可为'sum'/'mean'
def Slice(totalData, deviceList, metricList, user=None, device=None, timeRange=None, metric=None, groups=None,
          agg='sum'):
    data = pd.DataFrame()
    if user:
        for u in user:
            tmp = totalData[totalData['user'] == u]
            data = data.append(tmp)
    if device:
        if user:
            data1 = pd.DataFrame()
            for d in device:
                tmp = data[data['device'] == d]
                data1 = data1.append(tmp)
            data = data1
        else:
            for d in device:
                tmp = totalData[totalData['device'] == d]
                data = data.append(tmp)
    if not user and not device:
        data = totalData
    if timeRange:
        data = data[timeRange[0]:timeRange[1]]
    if metric:
        metricList = metric
        if not isinstance(metric, list):
            metric = [metric]
        data = data.loc[:, metric + ['device', 'user', 'month', 'date', 'time']]
    if groups:
        data = data.pivot_table(index=groups, values=metricList, aggfunc=agg)
    else:
        data = data.pivot_table(index=['device', 'time'], values=metricList)
    return data


# 钻取
# timeMode=0/1/2:时间维度为分钟/天/月
# zoneMode=0/1:空间维度为设备/用户
def Drill(totalData, deviceList, metricList, user=None, device=None, timeRange=None, metric=None, timeMode=0,
          zoneMode=0):
    data = pd.DataFrame()
    if user:
        for u in user:
            tmp = totalData[totalData['user'] == u]
            data = data.append(tmp)

    if device:
        if user:
            data1 = pd.DataFrame()
            for d in device:
                tmp = data[data['device'] == d]
                data1 = data1.append(tmp)
            data = data1
        else:
            for d in device:
                tmp = totalData[totalData['device'] == d]
                data = data.append(tmp)
    if not user and not device:
        data = totalData
    if timeRange:
        data = data[timeRange[0]:timeRange[1]]
    if metric:
        metricList = metric
        if not isinstance(metric, list):
            metric = [metric]
        data = data.loc[:, metric + ['device', 'user', 'month', 'date', 'time']]
    if timeMode == 0 and zoneMode == 0:
        return data
    groups = ['time', 'device']
    if timeMode == 1:
        groups[0] = 'date'
    elif timeMode == 2:
        groups[0] = 'month'
    if zoneMode == 1:
        groups[1] = 'user'
    data = data.pivot_table(index=groups, values=metricList, aggfunc='sum')
    return data


if __name__ == "__main__":
    dataDir = 'data\\常州天和印染有限公司\\'

    # 将整个建筑的信息放进一张大表。行为(时间+设备+用户),列为属性
    # totalData, deviceList, metricList = readData(dataDir)
    #
    pd.set_option('display.max_columns', None)
    # # totalData.to_csv('total.csv')
    # # 选项字段
    # user = ["常州天和印染有限公司"]
    # device = deviceList[2:5]
    # timeRange = ['2019-3-15', '2019-4-14']
    # metric = ['A相电压', '三相总有功功率']
    totalData, deviceList, metricList = Tool.olapData()
    user = [11]
    device = ['100001','100002']
    timeRange = ['2019-02-01', '2019-03-02']
    metric = ['BV','AA']
    '''切片和切块'''
    # 1.选定[用户+设备]切片
    # dataSlice7 = Slice(totalData, deviceList, metricList, user, device)
    # # 2.选定[用户+设备+时间段]切块
    # dataSlice2 = Slice(totalData, deviceList, metricList, user, device, timeRange)
    # # 3.选定[用户+设备+时间段+属性]切块
    # dataSlice7 = Slice(totalData, deviceList, metricList, user, device, timeRange, metric)
    # # 4.选定[用户+设备+时间段+属性]切块+按用户聚合[求和/求平均]
    # dataSlice4 = Slice(totalData, deviceList, metricList, user, device, timeRange, metric, ['date'], 'sum')
    # # 5.选定[用户+设备+时间段+属性]切块+按天聚合[求和/求平均]
    # dataSlice5 = Slice(totalData, deviceList, metricList, user, device, timeRange, metric, ['date'], 'sum')
    # # 6.选定[用户+设备+时间段+属性]切块+按设备聚合[求和/求平均]
    dataSlice7 = Slice(totalData, deviceList, metricList, user, device, timeRange, metric, ['device'], 'mean')
    # # 7.选定[用户+设备+时间段+属性]切块+按[设备+天]聚合[求和/求平均]
    # dataSlice7 = Slice(totalData, deviceList, metricList, user, device, timeRange, metric, ['device', 'date'], 'sum')
    dataSlice7.to_csv('dataSlice.csv')
    # print(dataSlice7)
    # # dataSlice1.to_csv('slice1.csv')
    for x in dataSlice7.columns.values.tolist():
        print(dataSlice7.loc[:, x].tolist())
    print(dataSlice7.index.name)
    print()

    # # 对聚合后数据做柱状图示例
    # dataSlice7.T.plot(kind='bar')
    # dataSlice7.plot(kind='bar')
    #
    # '''钻取：上钻or下钻'''
    # # 根据[用户+设备+时间段+属性]切块+选定时空维度的粒度进行上钻/下钻
    # # 粒度：分钟+设备
    # dataDrill1 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, 0, 0)
    # # 粒度：天+设备
    # dataDrill2 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, 1, 0)
    # # 粒度：月+设备
    # dataDrill3 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, 2, 0)
    # # 粒度：分钟+用户
    # dataDrill4 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, 0, 1)
    # # 粒度：天+用户
    # dataDrill5 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, 1, 1)
    # # 粒度：月+用户
    # dataDrill6 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, 2, 1)
    # print(dataDrill2.index)
    # print(isinstance(dataDrill2, MultiIndex))
    # dataDrill2.to_csv("datadrill6.csv")
    #

    # # 对上钻后数据做柱状图示例
    # dataDrill3.T.plot(kind='bar')
    #
    # '''旋转：行列互换'''
    # dataRotate = totalData.T
    # plt.show()



