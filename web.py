from flask import Flask, request
from flask_restful import Resource, Api
import hashlib
import pymysql
import threading
import json
import os
from usedMain import oldMain, correlation, train_forecast
from modelFunc import predictRealData, predictFunc, correlationFunc, clusterFunc, baseLine,profileFeatureFunc
from olap_code import Slice
from olap_code import Drill
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
        baseLine(factory, line, device, measurePoint, year, month, day)


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
        metric = js.get('metric')
        para1 = js.get('para1')
        para2 = js.get('para2')
        dataSlice7 = Slice(totalData, deviceList, metricList, user, device, timeRange, metric, para1,
                           para2)


class OlapDrill(Resource):
    def post(self):
        js: dict = request.json
        user = js.get('factory')
        device = js.get('device')
        timeRange = js.get('timeRange')
        metric = js.get('metric')
        para1 = js.get('para1')
        para2 = js.get('para2')
        dataDrill1 = Drill(totalData, deviceList, metricList, user, device, timeRange, metric, para1, para2)


api.add_resource(Predict, '/algorithm/predict')
api.add_resource(Correlation, '/algorithm/correlation')
api.add_resource(Cluster, '/algorithm/cluster')
api.add_resource(BaseLine, '/algorithm/baseline')
api.add_resource(ProfileFeature, '/algorithm/profilefeature')
api.add_resource(OlapSlice, '/algorithm/olapslice')
api.add_resource(OlapDrill, '/algorithm/olapdrill')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
