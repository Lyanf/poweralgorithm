import os
import pandas as pd
from tqdm import tqdm
import sqlalchemy
import pymysql
import datetime
import numpy as np
import pyodbc

class Tool:
    SHARED_ROOT = os.getenv("SHARED_ROOT")
    #SQL_HOST = "localhost"
    SQL_HOST = "121.46.233.22"
    __mapping = {}
    __mappingInit = False

    @classmethod
    def __initMapping(cls):
        if not cls.__mappingInit:
            cls.__mappingInit = True
            cls.__mapping["A相电压"] = "APhaseElectricTension"
            cls.__mapping["B相电压"] = "BPhaseElectricTension"
            cls.__mapping["C相电压"] = "CPhaseElectricTension"
            cls.__mapping["AB线电压"] = "ABLineElectricTension"
            cls.__mapping["BC线电压"] = "BCLineElectricTension"
            cls.__mapping["CA线电压"] = "CALineElectricTension"
            cls.__mapping["A相电流"] = "APhaseElectricCurrent"
            cls.__mapping["B相电流"] = "BPhaseElectricCurrent"
            cls.__mapping["C相电流"] = "CPhaseElectricCurrent"
            cls.__mapping["零序电流,"] = "ZeroSequenceCurrent"
            cls.__mapping["A相有功功率"] = "APhaseActivePower"
            cls.__mapping["B相有功功率"] = "BPhaseActivePower"
            cls.__mapping["C相有功功率"] = "CPhaseActivePower"
            cls.__mapping["三相总有功功率"] = "ThreePhaseTotalActivePower"
            cls.__mapping["A相无功功率"] = "APhaseReactivePower"
            cls.__mapping["B相无功功率"] = "BPhaseReactivePower"
            cls.__mapping["C相无功功率"] = "CPhaseReactivePower"
            cls.__mapping["三相总无功功率"] = "ThreePhaseTotalReactivePower"
            cls.__mapping["A相视在功率"] = "APhaseAtPower"
            cls.__mapping["B相视在功率"] = "BPhaseAtPower"
            cls.__mapping["C相视在功率"] = "CPhaseAtPower"
            cls.__mapping["三相总视在功率"] = "ThreePhaseTotalAtPower"
            cls.__mapping["A相功率因数"] = "APhasePowerFactor"
            cls.__mapping["B相功率因数"] = "BPhasePowerFactor"
            cls.__mapping["C相功率因数"] = "CPhasePowerFactor"
            cls.__mapping["平均功率因数"] = "AveragePowerFactor"
            cls.__mapping["频率"] = "Frequency"
            cls.__mapping["正向有功电度"] = "ForwardActive"
            cls.__mapping["反向有功电度"] = "ReverseActive"
            cls.__mapping["正向无功电度"] = "ForwardReactiveWattage"
            cls.__mapping["反向无功电度"] = "ReverseReactiveWattage"
            cls.__mapping["电压不平衡度"] = "VoltageUnbalance"
            cls.__mapping["电流不平衡度"] = "ElectricCurrentUnbalance"
            cls.__mapping["A相电压谐波总失真"] = "APhaseVoltageHarmonicTotalDistortion"
            cls.__mapping["B相电压谐波总失真"] = "BPhaseVoltageHarmonicTotalDistortion"
            cls.__mapping["C相电压谐波总失真"] = "CPhaseVoltageHarmonicTotalDistortion"
            cls.__mapping["A相电流谐波总失真"] = "TotalHarmonicDistortionOfAPhaseCurrent"
            cls.__mapping["B相电流谐波总失真"] = "TotalHarmonicDistortionOfBPhaseCurrent"
            cls.__mapping["C相电流谐波总失真"] = "TotalHarmonicDistortionOfCPhaseCurrent"
            cls.__mapping["正向有功最大需量"] = "MaximumPositiveActiveDemand"
            cls.__mapping["反向有功最大需量"] = "MaximumReverseActiveDemand"
            cls.__mapping["正向无功最大需量"] = "MaximumForwardReactivePowerDemand"
            cls.__mapping["反向无功最大需量"] = "MaximumReverseReactivePowerDemand"

    @classmethod
    def getMeasurePointMapping(cls):
        cls.__initMapping()
        return cls.__mapping

    @classmethod
    def getSQLEngine(cls):
        #db = sqlalchemy.create_engine("mysql+pymysql://root:dclab@%s/powersystem" % (cls.SQL_HOST))
        cnxn = pyodbc.connect("DSN=sjtudata")

        return cnxn

    @classmethod
    def excuteSQL(cls, sql):
        # db = pymysql.connect(host=cls.SQL_HOST, user="root", password="dclab", db="powersystem")
        # 使用 cursor() 方法创建一个游标对象 cursor
        db = Tool.getSQLEngine();
        db.setencoding('utf-8')
        cursor = db.cursor()

        # 使用 execute()  方法执行 SQL 查询，目的是把算出的结果进行保存，方便后面用户查到
        cursor.execute("SET character.literal.as.string=TRUE;")
        cursor.execute(sql)
        db.commit()

        cursor.close()
        db.close()

    # 指定一个测点，获取这个公司所有的设备，并且放进去，第一行放设备名称，然后返回第几列是选中的那个设备
    @staticmethod
    def getP_total(factory, line, device, measurePoint):
        dataDir = os.path.join(Tool.SHARED_ROOT, "data", factory)
        P_total = Tool.readData(dataDir, (measurePoint))
        if not os.path.exists(os.path.join(Tool.SHARED_ROOT, "data", "tmp")):
            os.mkdir(os.path.join(Tool.SHARED_ROOT, "data", "tmp"))
        P_total.to_csv(os.path.join(Tool.SHARED_ROOT, 'data', 'tmp', 'P_total.csv'))
        P_total.index = pd.to_datetime(P_total.index)
        # 补全device名并得到device_index
        for i in range(P_total.shape[1]):
            if device in P_total.columns[i]:
                device = P_total.columns[i]
                device_index = i
        if device_index == -1:
            raise NameError('Check the device!')
        return P_total, device_index

    @staticmethod
    def readData(dataDir, name):
        fileList = os.listdir(dataDir)
        P_total = pd.DataFrame()
        for file in tqdm(fileList):
            fileDir = os.path.join(dataDir, file)
            data = pd.read_excel(fileDir, header=1, index_col=0)
            data.index = pd.to_datetime(data.index)
            data = data['2019-03-06':'2019-05-13']
            if name in data.columns:
                tmp = pd.DataFrame(data[name]).rename(columns={name: file})
                P_total = P_total.join(tmp, how='outer')
        P_total.dropna(axis=0, how='any', inplace=True)
        P_total.drop_duplicates(inplace=True)
        return P_total

    @staticmethod
    def getMeasurePointMap():
        return

    @staticmethod
    def getP_totalBySQL(factory, device, measurePoint, timeRange = None):
        # SQLEngine = Tool.getSQLEngine()
        # deviceList = []
        # # for i in pd.read_sql("select distinct device from datas where factory='%s'" % (factory), SQLEngine).values:
        # for i in pd.read_sql("select distinct device from " + Tool.translateTable() +" where factory='%s'" % (factory), SQLEngine).values:
        #     deviceList.append(i[0])
        # deviceList.remove(device)
        # tableNames = "abcdefghijklmnopqrstuvwxyz"
        # fieldNames = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9""a10", "a11", "a12", "a13", "a14",
        #               "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12", "b13", "b14"]
        # # usedFieldNames = []
        # # for deviceName in deviceList:
        # baseSQL1 = "select basetable.timestamps,basetable.%s  " % ("basetable")
        # baseSQL2 = "  from (select timestamps,%s as %s from " + Tool.translateTable() +" where factory='%s' and line= '%s' and device = '%s') as basetable" \
        #            % (measurePoint, "basetable", factory, line, device)
        # selectFiled = ''
        # joinSQL = ''
        # # 前半段select
        # SQL1 = 'select timestamps'
        # # 后半段join的部分
        # SQL2 = ''
        # for i in range(0, len(deviceList)):
        #     fieldName = fieldNames[i]
        #     deviceName = deviceList[i]
        #     tableName = tableNames[i]
        #     # 以device为基准，增加其他设备的值，所以自身就不用加了
        #     selectFiled = selectFiled + " ,%s.%s" % (tableName, fieldName)
        #     joinSQL = joinSQL + " left outer join (select timestamps, %s as '%s' from " + Tool.translateTable() +" where factory = '%s' and device = '%s')as %s" \
        #                         " on basetable.timestamps = %s.timestamps " \
        #               % (measurePoint, fieldName, factory, deviceName, tableName, tableName)
        #
        # allSQL = baseSQL1 + selectFiled + baseSQL2 + joinSQL
        df = pd.read_sql('SELECT distinct meterid FROM rtdata;', Tool.getSQLEngine())
        sql = "select * from (select regdate AS timestamps "
        deviceList = []

        for i in range(len(df)):
            deviceList.append(df.iloc[i].values[0])

            sql += ",max(case meterid when '" + df.iloc[i].values[0] + "' then culunmvalue else 0 end) as '" + \
                   df.iloc[i].values[0] + "'"
        sql += " from rtdata WHERE metercolumn = '" + measurePoint + "'"
        if factory != "-1":
            sql += " and customerid = " + factory
        if timeRange != None:
            sql = sql + " and regdate > DATE('" + timeRange[0] + "') AND regdate <= DATE('" + timeRange[1] + "') "
        sql += " group by regdate)"

        originDataFrame = pd.read_sql(sql, Tool.getSQLEngine(), "timestamps")
        indexedDataFrame = originDataFrame
        # print(allSQL)
        # originDataFrame = pd.read_sql(allSQL, SQLEngine)
        # indexedDataFrame = originDataFrame.set_index("timestamps")
        indexedDataFrame.index = pd.to_datetime(indexedDataFrame.index)
        indexedDataFrame = indexedDataFrame.sort_index()

        # 指定了时间
        # indexedDataFrame = indexedDataFrame['2019-03-06':'2019-05-13']
        indexedDataFrame = indexedDataFrame.dropna()
        indexedDataFrame = indexedDataFrame.drop_duplicates()
        # deviceList.insert(0, device)
        indexedDataFrame.columns = deviceList

        del_list = []
        for i in range(indexedDataFrame.shape[1]):
            if max(indexedDataFrame.iloc[:, i]) - min(indexedDataFrame.iloc[:, i]) == 0:
                del_list.append(i)
        indexedDataFrame.drop(indexedDataFrame.columns[del_list], axis=1, inplace=True)
        # 返回一个0是因为，原本的P_total需要一个device_index标记当前选择的是哪个设备

        if device == "-1":
            return indexedDataFrame, -1
        reloc = 0
        for i in indexedDataFrame.columns.values.tolist():
            reloc += 1
            if i == device:
                break
        if reloc == 0:
            raise Exception("所选设备在该测点数据异常")

        return indexedDataFrame, reloc


    @staticmethod
    def getData(data, date, delta,day_point):
        date -= datetime.timedelta(days=delta)
        dateStr = str(date.year) + '-' + "%0.2d"%(date.month)+ '-' + "%0.2d"%(date.day)
        # dateStr = str(date.year) + '-' + str(date.month) + '-' + str(date.day)
        res = data[dateStr]
        count = 0
        while len(res) != day_point:
            date -= datetime.timedelta(days=1)
            dateStr = str(date.year) + '-' + "%0.2d"%(date.month)+ '-' + "%0.2d"%(date.day)
            # dateStr = str(date.year) + '-' + str(date.month) + '-' + str(date.day)
            res = data[dateStr]
            count += 1
            if count > 10:
                return 0
        return np.array(res)

    # @staticmethod
    # def olapReadDataBySQL(dataDir):
    #     fileList = os.listdir(dataDir)
    #     res = pd.DataFrame()
    #     for file in fileList:
    #         fileDir = dataDir + file
    #         data = pd.read_excel(fileDir, header=1, index_col=0)
    #         data.index = pd.to_datetime(data.index)
    #         data['date'] = [datetime.datetime.strftime(x, '%Y-%m-%d') for x in data.index]
    #         data['month'] = [x.month for x in data.index]
    #         data['device'] = file
    #         data['user'] = "常州天和印染有限公司"
    #         res = res.append(data)
    #     res['time'] = res.index
    #     metricList = list(res.columns)
    #     metricList.remove('user')
    #     metricList.remove('month')
    #     metricList.remove('date')
    #     metricList.remove('device')
    #     metricList.remove('time')
    #     return res, fileList, metricList

    @staticmethod
    def translateTable(timeRange: list = None):
        SQLEngine = Tool.getSQLEngine()
        sql = "SELECT distinct metercolumn FROM rtdata"
        metercolumnlist = SQLEngine.cursor().execute(sql).fetchall()
        sqltable = "select regdate as timestamps, customerid as factory, meterid AS device"
        for m in metercolumnlist:
        	sqltable =  sqltable + ", max(case metercolumn when '" +m[0] + "' then culunmvalue else 0 end) as " + m[0]
        sqltable = sqltable + " from rtdata "
        if timeRange != None:
            sqltable  = sqltable + " where regdate > DATE('" + timeRange[0] + "') AND regdate <= DATE('" + timeRange[1] + "') "
        sqltable = sqltable + " group by regdate,customerid ,meterid"
        sqltable = " (" +sqltable +") "

        return sqltable

    @staticmethod
    def getDevice():
        SQLEngine = Tool.getSQLEngine()
        sql = "SELECT distinct meterid FROM rtdata"
        meteridlist = SQLEngine.cursor().execute(sql).fetchall()
        mlist = [i[0] for i in meteridlist]
        SQLEngine.close()
        return mlist

    @staticmethod
    def getMeasurePoint():
        SQLEngine = Tool.getSQLEngine()
        sql = "SELECT distinct metercolumn FROM rtdata"
        metercolumnlist = SQLEngine.cursor().execute(sql).fetchall()
        mlist = [i[0] for i in metercolumnlist]
        SQLEngine.close()
        return mlist

    @staticmethod
    def olapData(timeRange: list = None):
        devicelist = Tool.getDevice()
        metriclist = Tool.getMeasurePoint()
        data = pd.read_sql(Tool.translateTable(timeRange), Tool.getSQLEngine(), index_col='timestamps')
        data.index = pd.to_datetime(data.index)
        column = ['user','device']
        column.extend(metriclist)
        data.columns = column
        data['date'] = [datetime.datetime.strftime(x, '%Y-%m-%d') for x in data.index]
        data['month'] = [x.month for x in data.index]

        data['time'] = data.index
        return data, devicelist, metriclist