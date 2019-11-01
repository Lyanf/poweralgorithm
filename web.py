from flask import Flask, request
from flask_restful import Resource, Api
import hashlib
import pymysql
import threading
import json
import os
from oriCode import oldMain, correlation, train_forecast
from modelFunc import predictFunc, correlationFunc, clusterFunc, baseLine

app = Flask(__name__)
api = Api(app)
allThread = []


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
        month = int(request.json['month'])+1
        day = int(request.json['day'])+1
        baseLine(factory, line, device, measurePoint, year, month, day)


api.add_resource(Predict, '/algorithm/predict')
api.add_resource(Correlation, '/algorithm/correlation')
api.add_resource(Cluster, '/algorithm/cluster')
api.add_resource(BaseLine, '/algorithm/baseline')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
