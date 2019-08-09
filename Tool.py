import os
import pandas as pd
from tqdm import tqdm
import sqlalchemy

class Tool:
    sharedroot = os.getenv("SHARED_ROOT")
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
    # 指定一个测点，获取这个公司所有的设备，并且放进去，第一行放设备名称，然后返回第几列是选中的那个设备
    @staticmethod
    def getP_total(factory, line, device, measurePoint):
        dataDir = os.path.join(Tool.sharedroot, "data", factory)
        P_total = Tool.readData(dataDir, (measurePoint))
        if not os.path.exists(os.path.join(Tool.sharedroot, "data", "tmp")):
            os.mkdir(os.path.join(Tool.sharedroot, "data", "tmp"))
        P_total.to_csv(os.path.join(Tool.sharedroot, 'data', 'tmp', 'P_total.csv'))
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
    def getP_totalBySQL(factory, line, device, measurePoint):
        db = sqlalchemy.create_engine("mysql+pymysql://root:dclab@localhost/powersystem")
        deviceList = []
        for i in pd.read_sql("select distinct device from datas where factory='%s'" % (factory), db).values:
            deviceList.append(i[0])
        deviceList.remove(device)
        tableNames = "abcdefghijklmnopqrstuvwxyz"
        fieldNames = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9""a10", "a11", "a12", "a13", "a14",
                      "b1","b2","b3","b4","b5","b6","b7","b8","b9","b10","b11","b12","b13","b14"]
        # usedFieldNames = []
        # for deviceName in deviceList:
        baseSQL1 = "select basetable.timestamp,basetable.%s  " % ("basetable")
        baseSQL2 = "  from (select timestamp,%s as %s from datas where factory='%s' and line= '%s' and device = '%s') as basetable" \
                   % (measurePoint, "basetable", factory, line, device)
        selectFiled = ''
        joinSQL = ''
        # 前半段select
        SQL1 = 'select timestamp'
        # 后半段join的部分
        SQL2 = ''
        for i in range(0, len(deviceList)):
            fieldName = fieldNames[i]
            deviceName = deviceList[i]
            tableName = tableNames[i]
            # 以device为基准，增加其他设备的值，所以自身就不用加了
            selectFiled = selectFiled + " ,%s.%s" % (tableName, fieldName)
            joinSQL = joinSQL + " left outer join (select timestamp, %s as '%s' from datas where factory = '%s' and device = '%s')as %s" \
                                " on basetable.timestamp = %s.timestamp " \
                      % (measurePoint, fieldName, factory, deviceName, tableName, tableName)

        allSQL = baseSQL1 + selectFiled + baseSQL2 + joinSQL
        originDataFrame = pd.read_sql(allSQL, db)
        indexedDataFrame = originDataFrame.set_index("timestamp")
        indexedDataFrame.index = pd.to_datetime(indexedDataFrame.index)
        # 指定了时间
        indexedDataFrame = indexedDataFrame['2019-03-06':'2019-05-13']
        indexedDataFrame = indexedDataFrame.dropna()
        indexedDataFrame = indexedDataFrame.drop_duplicates()
        deviceList.insert(0, device)
        indexedDataFrame.columns = deviceList
        # 返回一个0是因为，原本的P_total需要一个device_index标记当前选择的是哪个设备
        # 但是新方法这个设备一定是在第0列
        return indexedDataFrame,0
