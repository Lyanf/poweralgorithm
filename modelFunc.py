import hashlib
import json
import os
import pymysql
from Tool import Tool
from oriCode import correlation, train_forecast


# 完全自己做，并且把数据放到数据库里面，把结果以json的格式防在文件夹里
def predictFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()
    resultFileName = parameterHash[0:15]

    # 先把结果的地址存到数据库里面，结果再慢慢后面算，目的是为了让同样的请求只能启动一个计算任务
    db = pymysql.connect(host="139.199.36.137", user="root", password="dclab", db="powersystem")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询，目的是把算出的结果进行保存，方便后面用户查到
    print(parameterHash, resultFileName)
    cursor.execute('''insert into powersystem.algorithmresult values("%s","%s")''' % (parameterHash, resultFileName))
    db.commit()

    P_total, device_index = Tool.getP_total(factory, line, device, measurePoint)
    corr_device = correlation(P_total, device_index, 3)
    a, b = train_forecast(P_total, corr_device, device_index)
    dirPath = os.path.join(Tool.sharedroot,"predict")
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    with open(os.path.join(dirPath, resultFileName + ".json"), "w") as f:
        json.dump({'y_true': a, 'y_pred': b}, f)


def correlationFunc(factory, line, device, measurePoint):
    allString = factory + line + device + measurePoint
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    parameterHash = md.hexdigest()


    P_total, device_index = Tool.getP_total(factory, line, device, measurePoint)
    corr_device = correlation(P_total, device_index, 3)
    # keys:带有设备的文件名  values:相似度
    a,b =  corr_device.keys(),corr_device.values
    resultDict = {}
    for i,j in zip(a,b):
        resultDict[i] = j

    resultJson = json.dumps(resultDict,ensure_ascii=False)
    db = pymysql.connect(host="139.199.36.137", user="root", password="dclab", db="powersystem")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询，目的是把算出的结果进行保存，方便后面用户查到
    sql = '''insert into powersystem.correlation values('%s','%s')''' % (parameterHash, resultJson)
    cursor.execute(sql)
    db.commit()