# -*-coding:utf-8-*-
from flask import Flask, request, Response, abort
from flask_cors import CORS
import time
import sys
import json
import traceback
import synonyms

app = Flask(__name__)
CORS(app)  # 允许所有路由上所有域使用CORS


class SynonymsModel:
    def load_model(self):
        self.syn = synonyms.nearby

    def generate_result(self, word):
        return self.syn(word)[0]


@app.route("/", methods=['POST', 'GET'])
def index():
    return '分词程序正在运行中'


@app.route("/synonyms", methods=['POST', 'GET'])
def get_result():
    if request.method == 'POST':
        word = request.data.decode("utf-8")
    else:
        word = request.args['text']

    try:
        start = time.time()
        print("用户输入", word)
        res = synonymsModel.generate_result(word)
        end = time.time()
        print('耗时：', end - start)
        result = {'code': '200', 'msg': '响应成功', 'data': res}
    except Exception as e:
        print(e)
        result_error = {'errcode': -1}
        result = json.dumps(result_error, indent=4, ensure_ascii=False)
        # 这里用于捕获更详细的异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        # 提前退出请求
        abort(Response("Failed!\n" + '\n\r\n'.join('' + line for line in lines)))
    return Response(str(result), mimetype='application/json')


if __name__ == "__main__":
    synonymsModel = SynonymsModel()
    synonymsModel.load_model()
    app.run(host='0.0.0.0', port=1314, threaded=False)
