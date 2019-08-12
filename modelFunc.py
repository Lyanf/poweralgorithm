import hashlib
import numpy as np
import json
import os
import pymysql
from Tool import Tool
from oriCode import correlation, train_forecast, cluster


# 完全自己做，并且把数据放到数据库里面，把结果以json的格式防在文件夹里
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
    hourList, dayList = cluster(np.array(P_total.iloc[:,device_index]))
    hourX = len(hourList[0])
    dayX = len(dayList[0])
    resultDict = {'hourX':list(range(0,hourX)),'dayX':list(range(0,dayX)),'hourList':hourList,'dayList':dayList}
    try:
        resultJson = json.dumps(resultDict,ensure_ascii=False)
    except Exception as e:
        e.with_traceback()

    sql ="insert into cluster (hash,json)values ('%s','%s')"%(parameterHash,resultJson)
    Tool.excuteSQL(sql)
