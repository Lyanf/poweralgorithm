
import hashlib
import numpy as np
import json
import os
import pymysql
import pyodbc
import pandas as pd
from Tool import Tool
from usedMain import correlation, train_forecast, cluster, profileFeature, baseline
import datetime
from olap_code import Slice, Drill



def predictRealData(factory, line, device, measurePoint, year, month, day):
    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    P_total = P_total[year + '-' + month + '-' + day]
    day_point = 480  # 一天为480个数据点
    P_forecast = P_total.iloc[:, device_index]
    y_total = P_forecast[day_point * 7:].reset_index(drop=True)
    return np.array(y_total)[-7 * day_point:].tolist()


# 完全自己做，并且把数据放到数据库里面
def predictFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()
    resultFileName = parameterHash[0:15]

    # insertSQL = '''insert into sjtudb.algorithmresult values('%s','%s',null)''' % (parameterHash, resultFileName)
    # Tool.excuteSQL(insertSQL)

    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)

    corr_device = correlation(P_total, device_index, 3)
    a, b = train_forecast(P_total, corr_device, device_index,96)
    lastResult = {'y_true': a, 'y_pred': b}
    jsonStr = json.dumps(lastResult)

    insertSQL = '''insert into sjtudb.algorithmresult values('%s','%s','%s')''' % (parameterHash, resultFileName,jsonStr)
    # print(insertSQL)
    Tool.excuteSQL(insertSQL)
    # updateSQL = "update sjtudb.algorithmresult set json='%s' where hash='%s'" % (jsonStr, parameterHash)
    # Tool.excuteSQL(updateSQL)

    # dirPath = os.path.join(Tool.sharedroot,"predict")
    # if not os.path.exists(dirPath):
    #     os.makedirs(dirPath)
    # with open(os.path.join(dirPath, resultFileName + ".json"), "w") as f:
    #     json.dump({'y_true': a, 'y_pred': b}, f)


def correlationFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()
    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    corr_device = correlation(P_total, device_index, 3)

    addSelfDeviceCorrIndex = corr_device.index.tolist()
    addSelfDeviceCorrIndex.append(device)
    corrData = P_total.loc[:, addSelfDeviceCorrIndex]

    corrDataDict = {
        'timestamp': corrData.index.strftime("%Y-%m-%d %H:%M:%S").values.tolist(),
    }
    for i in addSelfDeviceCorrIndex:
        corrDataDict[i] = corrData.loc[:, i].values.tolist()
    corrDataJson = json.dumps(corrDataDict, ensure_ascii=False)
    # keys:带有设备的文件名  values:相似度
    a, b = corr_device.keys(), corr_device.values
    resultDict = {}
    for i, j in zip(a, b):
        resultDict[i] = j

    resultJson = json.dumps(resultDict, ensure_ascii=False)


    sql = ''' insert into sjtudb.correlation values('%s','%s','%s') ''' % (parameterHash, resultJson, corrDataJson)
    Tool.excuteSQL(sql)


def clusterFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()

    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)

    kmeans_hour, labels_hour, kmeans_day, labels_day = cluster(np.array(P_total.iloc[:, device_index]), 96)

    hourList = kmeans_hour.tolist()
    dayList = kmeans_day.tolist()
    # print(dayList)
    hourX = len(hourList[0])
    dayX = len(dayList[0])
    resultDict = {'hourX': list(range(0, hourX)), 'dayX': list(range(0, dayX)), 'hourList': hourList,
                  'dayList': dayList}

    try:

        resultJson = json.dumps(resultDict, ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql = "insert into clusterresult (hashstr,json)values ('%s', '%s')" % (parameterHash, resultJson)
    Tool.excuteSQL(sql)


def baseLine(factory, line, device, measurePoint, year, month, day, day_point = 96):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()

    data, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    data = data.iloc[:, device_index]
    baseValue, trueValue = baseline(data, year, month, day, day_point)

    resultDict = {'baseValue': list(baseValue),
                  'trueValue': list(trueValue)}
    try:
        resultJson = json.dumps(resultDict, ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql = "insert into baseline (hashstr,json)values ('%s','%s')" % (parameterHash, resultJson)

    Tool.excuteSQL(sql)


def profileFeatureFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()

    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    kmeans_hour, labels_hour, kmeans_day, labels_day = cluster(np.array(P_total.iloc[:, device_index]), 96)

    temp8760 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ShanghaiTemp8760.csv'), header=None, sep="[;,]", engine='python')

    staticFeatures, dynamicFeatures, tempload, temp = profileFeature(P_total.iloc[:, device_index], kmeans_hour, kmeans_day,
                                                     labels_hour, labels_day, temp8760)

    staticFeatures[0]
    staticFeatures[5] = str(staticFeatures[5])[1:-2]
    staticFeatures[7] = staticFeatures[7].tolist()
    staticFeatures[8] = staticFeatures[8].tolist()
    dynamicFeatures[0] = dynamicFeatures[0].tolist()
    dynamicFeatures[2] = dynamicFeatures[2].tolist()
    staticFeaturesObj = {
        'maxv':staticFeatures[0],
        'minv':staticFeatures[1],
        'median': staticFeatures[2],
        'avg': staticFeatures[3],
        'standard': staticFeatures[4],
        'fftavg': staticFeatures[5],
        'fftstandard': staticFeatures[6],
        'featurelineh': staticFeatures[7], #典型特征模式曲线（小时尺度）
        'linearfeaturelined': staticFeatures[8], #线性特征模式曲线（天尺度）
    }
    dynamicFeaturesObj = {
        'transfermatrixh':dynamicFeatures[0], #基于聚类结果的马尔科夫转移矩阵（小时尺度）
        'entropyh':dynamicFeatures[1],
        'transfermatrixd':dynamicFeatures[2], #基于聚类结果的科尔科夫转移矩阵（天尺度）
        'entropyd':dynamicFeatures[3]
    }
    resultDict = {'static': staticFeaturesObj, 'dynamic': dynamicFeaturesObj, "load": tempload.tolist(), "temp": temp.tolist()}

    try:
        resultJson = json.dumps(resultDict,ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql = "insert into profilefeature (hashstr,json)values ('%s','%s')" % (parameterHash, resultJson)
    Tool.excuteSQL(sql)

def olapSlice(user = None,device  = None,timeRange  = None, metric  = None, groups:list  = None, agg = None, rotate = None):
    totalData, deviceList, metricList = Tool.olapData(timeRange)
    print(len(totalData))
    dataSlice = pd.DataFrame()
    if timeRange == None and metric == None and groups == None:
        # 1.选定[用户+设备]切片
        dataSlice = Slice(totalData, deviceList, metricList, user, device)
    elif timeRange != None and metric == None and groups == None:
        # 2.选定[用户+设备+时间段]切块
        dataSlice = Slice(totalData, deviceList, metricList, user, device, timeRange)
    elif timeRange != None and metric != None and groups == None:
        # 3.选定[用户+设备+时间段+属性]切块
        dataSlice = Slice(totalData, deviceList, metricList, user, device, timeRange, metric)
    elif timeRange != None and metric != None and groups != None:
        dataSlice = Slice(totalData, deviceList, metricList, user, device, timeRange, metric, groups, agg)

    print("slice")
    if rotate == None:
        dataSlice.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slice.csv'))
    return dataSlice


def olapDrill(user=None, device=None, timeRange=None, metric=None,timeMode = 0,zoneMode = 0):

    totalData, deviceList, metricList = Tool.olapData(timeRange)
    dataDrill = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, timeMode, zoneMode)
    print("drill")
    dataDrill.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drill.csv"))
    return
def olapRotate(user = None,device  = None,timeRange  = None, metric  = None, groups:list  = None, agg = None):
    data = olapSlice(user, device, timeRange, metric,groups, agg, 1)
    data.T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotate.csv"))
