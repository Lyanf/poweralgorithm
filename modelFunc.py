import hashlib
import numpy as np
import json
import os
import pymysql
from Tool import Tool
from usedMain import correlation, train_forecast, cluster, profileFeature
import datetime


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

    insertSQL = '''insert into powersystem.algorithmresult values("%s","%s",null)''' % (parameterHash, resultFileName)
    Tool.excuteSQL(insertSQL)

    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    corr_device = correlation(P_total, device_index, 3)
    a, b = train_forecast(P_total, corr_device, device_index)
    lastResult = {'y_true': a, 'y_pred': b}
    jsonStr = json.dumps(lastResult)
    updateSQL = "update powersystem.algorithmresult set json='%s' where hash='%s'" % (jsonStr, parameterHash)
    Tool.excuteSQL(updateSQL)

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

    sql = '''insert into powersystem.correlation values('%s','%s','%s')''' % (parameterHash, resultJson, corrDataJson)
    Tool.excuteSQL(sql)


def clusterFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()

    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    kmeans_hour, labels_hour, kmeans_day, labels_day = cluster(np.array(P_total.iloc[:, device_index]))
    hourList = kmeans_hour
    dayList = kmeans_day
    hourX = len(hourList[0])
    dayX = len(dayList[0])
    resultDict = {'hourX': list(range(0, hourX)), 'dayX': list(range(0, dayX)), 'hourList': hourList,
                  'dayList': dayList}
    try:
        resultJson = json.dumps(resultDict, ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql = "insert into cluster (hash,json)values ('%s','%s')" % (parameterHash, resultJson)
    Tool.excuteSQL(sql)


def baseLine(factory, line, device, measurePoint, year, month, day):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()

    data, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    data = data.iloc[:, device_index]

    date = datetime.datetime(year, month, day)
    date1 = date - datetime.timedelta(days=1)
    date2 = date - datetime.timedelta(days=2)
    date3 = date - datetime.timedelta(days=3)
    date7 = date - datetime.timedelta(days=7)

    data1, data2, data3, data7 = Tool.getData(data, date, 1), Tool.getData(data, date, 2), Tool.getData(data, date,
                                                                                                        3), Tool.getData(
        data, date, 7)

    res = (data1 + data2 + data3 + data7) / 4  # 该设备该日期的能耗基线

    # plt.figure(figsize=(16, 8))
    # plt.plot(res, label="能耗基线")
    # plt.plot(np.array(data[str(date.year) + '-' + str(date.month) + '-' + str(date.day)]), label="实际值")
    # plt.legend()
    # plt.title('能耗基线与实际值对比')
    baseValue = res
    trueValue = np.array(data[str(date.year) + '-' + str(date.month) + '-' + str(date.day)])

    resultDict = {'baseValue': list(baseValue),
                  'trueValue': list(trueValue)}
    try:
        resultJson = json.dumps(resultDict, ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql = "insert into baseline (hash,json)values ('%s','%s')" % (parameterHash, resultJson)
    Tool.excuteSQL(sql)


def profileFeatureFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()

    P_total, device_index = Tool.getP_totalBySQL(factory, line, device, measurePoint)
    kmeans_hour, labels_hour, kmeans_day, labels_day = cluster(np.array(P_total.iloc[:, device_index]))
    staticFeatures, dynamicFeatures = profileFeature(P_total.iloc[:, device_index], kmeans_hour, kmeans_day,
                                                     labels_hour, labels_day)
    staticFeatures[0]
    staticFeatures[5] = str(staticFeatures[5])
    staticFeatures[7] = staticFeatures[7].tolist()
    staticFeatures[8] = staticFeatures[8].tolist()
    dynamicFeatures[0] = dynamicFeatures[0].tolist()
    dynamicFeatures[2] = dynamicFeatures[2].tolist()
    staticFeaturesObj = {
        '最大值':staticFeatures[0],
        '最小值':staticFeatures[1],
        '中位数': staticFeatures[2],
        '均值': staticFeatures[3],
        '标准差': staticFeatures[4],
        'fft频谱均值': staticFeatures[5],
        'fft频谱标准差': staticFeatures[6],
        '典型特征模式曲线（小时尺度）': staticFeatures[7],
        '线性特征模式曲线（天尺度）': staticFeatures[8],
    }
    dynamicFeaturesObj = {
        '基于聚类结果的马尔科夫转移矩阵（小时尺度）':dynamicFeatures[0],
        '行为信息熵（小时尺度）':dynamicFeatures[1],
        '基于聚类结果的科尔科夫转移矩阵（天尺度）':dynamicFeatures[2],
        '行为信息熵（天尺度）':dynamicFeatures[3]
    }
    resultDict = {'静态特性': staticFeaturesObj, '动态特性': dynamicFeaturesObj}

    try:
        resultJson = json.dumps(resultDict,ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql = "insert into profileFeature (hash,json)values ('%s','%s')" % (parameterHash, resultJson)
    Tool.excuteSQL(sql)

def olapSlice(totalData,deviceList,metricList,user,device,timeRange,metric,collect:list,method):
    pass

def olapDrill():
    pass
