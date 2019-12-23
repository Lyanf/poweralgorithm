from flask import Flask, request
from flask_restful import Resource, Api
import hashlib
import pymysql
import threading
import json
import os
from usedMain import oldMain, correlation, train_forecast
from modelFunc import predictRealData, predictFunc, correlationFunc, clusterFunc, baseLine,profileFeatureFunc, olapSlice, olapDrill, olapRotate
from olap_code import Slice
from olap_code import Drill

from Tool import Tool

app = Flask(__name__)
api = Api(app)
allThread = []


class PredictRealData(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        measurePoint = request.json['measurePoint']
        year = int(request.json['year'])
        month = int(request.json['month'])
        day = int(request.json['day'])
        allString = factory + line + device + measurePoint
        print(allString)
        assert isinstance(allString, str)

        print(factory, line, device, measurePoint)
        return predictRealData(factory, line, device, measurePoint, year, month, day)


class Predict(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        measurePoint = request.json['measurePoint']
        allString = factory + line + device + measurePoint
        print(allString)
        assert isinstance(allString, str)

        print(factory, line, device, measurePoint)
        predictFunc(factory, line, device, measurePoint)

        return {'status': "ok"}

    def get(self):
        print("this is get method!")


class Correlation(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        measurePoint = request.json['measurePoint']
        print(factory, line, device, measurePoint)
        correlationFunc(factory, line, device, measurePoint)
        return {'status': "ok"}


class Cluster(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        measurePoint = request.json['measurePoint']
        print(factory, line, device, measurePoint)
        clusterFunc(factory, line, device, measurePoint)
        return {'status': "ok"}


class BaseLine(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        measurePoint = request.json['measurePoint']
        print(factory, line, device, measurePoint)
        year = int(request.json['year'])
        month = int(request.json['month'])
        day = int(request.json['day'])
        baseLine(factory, line, device, measurePoint, year, month, day, 96)


class ProfileFeature(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        measurePoint = request.json['measurePoint']
        print(factory, line, device, measurePoint)
        profileFeatureFunc(factory, line, device, measurePoint)

class OlapSlice(Resource):
    def post(self):
        js:dict = request.json
        user =  js.get('factory')
        device = js.get('device')
        timeRange = js.get('timeRange')
        metric = js.get('measurePoint')
        group = js.get('group')
        agg = js.get('agg')
        if len(device) == 0:
            device = None
        if len(timeRange) == 0:
            timeRange = None
        if len(metric) == 0:
            metric = None
        if len(group) == 0:
            group = None
        if len(agg) == 0:
            agg = None
        print(timeRange)
        print(metric)
        print(group)
        print(agg)
        dataSlice7 = olapSlice(user, device, timeRange, metric,group, agg)


class OlapDrill(Resource):
    def post(self):
        js: dict = request.json
        user = js.get('factory')
        device = js.get('device')
        timeRange = js.get('timeRange')
        metric = js.get('measurePoint')
        timeMode = js.get('timeMode')
        zoneMode = js.get('zoneMode')
        if len(user) == 0:
            user = None
        if len(device) == 0:
            device = None
        if len(timeRange) == 0:
            timeRange = None
        if len(metric) == 0:
            metric = None
        print(device)
        print(metric)
        print(zoneMode)
        print(timeRange)
        print(timeMode)
        dataDrill1 = olapDrill(user, device, timeRange, metric, timeMode, zoneMode)


class OlapRotate(Resource):
    def post(self):
        js: dict = request.json
        user = js.get('factory')
        device = js.get('device')
        timeRange = js.get('timeRange')
        metric = js.get('metric')
        group = js.get('group')
        agg = js.get('agg')

        rotate = olapRotate(user, device, timeRange, metric,group, agg)


class Test(Resource):
    def get(self):
        Tool.translateTable()

api.add_resource(Predict, '/algorithm/predict')
api.add_resource(Correlation, '/algorithm/correlation')
api.add_resource(Cluster, '/algorithm/cluster')
api.add_resource(BaseLine, '/algorithm/baseline')
api.add_resource(ProfileFeature, '/algorithm/profilefeature')
api.add_resource(OlapSlice, '/algorithm/olapslice')
api.add_resource(OlapDrill, '/algorithm/olapdrill')
api.add_resource(OlapRotate, '/algorithm/olaprotate')

api.add_resource(Test,'/test')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
