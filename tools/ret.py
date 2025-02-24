# -*- coding: utf-8 -*-
import json


class JsonRet(object):
    RET_OK = 0
    RET_FAIL = 1
    RET_ERROR_PARAM = 2
    RET_NOT_EXIST = 3
    RET_WECHAT_ERROR = 105001
    RET_WECHAT_RETURN_FAIL = 105002

    def __init__(self):
        self.code = self.RET_OK
        self.msg = ""
        self.data = None

    def set_data(self, data):
        self.data = data

    def set_msg(self, msg):
        self.msg = msg

    def set_code(self, code):
        self.code = code

    def __call__(self, code=None, msg=None, data=None):
        if code is not None:
            self.set_code(code)
        if msg is not None:
            self.set_msg(msg)
        if data is not None:
            self.set_data(data)
        if self.data is None:
            ret_dic = {
                "code": self.code,
                "msg": self.msg
            }
        else:
            ret_dic = {
                "code": self.code,
                "msg": self.msg,
                "value": self.data
            }
        return json.dumps(ret_dic)
