"""
测试起的服务
"""
import requests


def test_get_words():
    url = "http://192.168.0.101:1314/synonyms"
    data = "北京"
    data = data.encode('utf-8')
    res = requests.post(url, data=data)
    print(res.text)


if __name__ == '__main__':
    test_get_words()
