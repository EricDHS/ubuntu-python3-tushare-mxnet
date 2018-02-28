# -*- coding: utf-8 -*-
import urllib
import re
import math
import sys
import os
import threading
import datetime
import tushare as ts
import threading
import time
import logging
import subprocess

f = open('config/pure_code', 'r')
logger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s|%(levelname)s - %(threadName)s - %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.INFO)
logger.addHandler(consoleHandler)
access_h = logging.FileHandler('{}/{}-access.log'.format('log', subprocess.check_output('hostname', shell=True).strip().decode('utf-8')))
access_h.setFormatter(logFormatter)
access_h.setLevel(logging.INFO)
logger.addHandler(access_h)
error_h = logging.FileHandler('{}/{}-error.log'.format('log', subprocess.check_output('hostname', shell=True).strip().decode('utf-8')))
error_h.setFormatter(logFormatter)
error_h.setLevel(logging.ERROR)
logger.addHandler(error_h)
logger.setLevel(logging.INFO)

threadLock = threading.Lock()
threads = []
class myThread (threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
    def run(self):
        logger.info("Starting " + self.name)
        while True:
            code = get_code()
            if not code:
                return
            try:
                ts.get_hist_data(code).to_csv('data/hist/{}_hist.csv'.format(code))
                logger.info('Successfully fetched stock={} data'.format(code))
            except Exception as e:
                print('Error occured: {}, code: {}'.format(e, code))

def get_code():
    threadLock.acquire()
    r = f.readline()
    threadLock.release()
    return r.strip()

# 创建新线程
i = 0
while (i<1):
    threads.append(myThread(i))
    i = i + 1 


# 开启新线程
for thread in threads:
    thread.start()

# 等待所有线程完成
for t in threads:
    t.join()

f.close()
