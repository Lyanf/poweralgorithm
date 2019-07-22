from flask import Flask,request
from flask_restful import Resource,Api
import hashlib
import pymysql
import threading
import json
import os
from main import oldMain
app = Flask(__name__)
api = Api(app)
allThread = []
sharedRoot = os.getenv("SHARED_ROOT")
def predictFunc(factory,line,device):
    allString = factory + line + device
    assert isinstance(allString, str)
    md = hashlib.md5()
    md.update(allString.encode("utf8"))
    hash = md.hexdigest()
    result = hash[0:15]
    print(type(hash))
    print(factory, line, device)
    db = pymysql.connect(host="139.199.36.137", user="root", password="dclab", db="powersystem")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    print(hash,result)
    cursor.execute('''insert into algorithmresult values("%s","%s")'''%(hash,result))

    db.commit()
    a, b = oldMain(factory, line, device)
    lastResultJson =  {'y_true': a, 'y_pred': b}
    with open(os.path.join(sharedRoot,result+".json"),"w") as f:
        json.dump({'y_true':a,'y_pred':b},f)
class Predict(Resource):
    def post(self):
        factory = request.json['factory']
        line = request.json['line']
        device = request.json['device']
        allString = factory+line+device
        print(allString)
        assert isinstance(allString,str)

        print(factory,line,device)
        # tempThread = threading.Thread(target=predictFunc,args=(factory,line,device))
        # tempThread.start()
        # allThread.append(tempThread)
        predictFunc(factory,line,device)

        return {'status':"ok"}
    def get(self):
        print("this is get method!")
api.add_resource(Predict,'/algorithm/predict')
if __name__ == '__main__':
    app.run(debug=True)