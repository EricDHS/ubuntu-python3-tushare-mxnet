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
_BASICS_ = 'basics_report'
_YEJI_ = 'yeji_report'
_PROFIT_ = 'profit_report'
_OPERATION_= 'operation_report'
_GROWTH_ = 'grown_report'
_DEBTPAYING_ = 'debat_paying_report'
_CASHFLOW_ = 'cash_flow_report'

class myThread (threading.Thread):
    def __init__(self, name, code):
        threading.Thread.__init__(self)
        self.name = name
        self.code = code
    def run(self):
        logger.info("Starting name: {}, merging data for code: {}".format(self.name, self.code))
        try:
            hist_data = pd.read_csv('data/hist/{}_hist.csv'.format(self.code)).iloc[::-1]
            for 
            file_name = 'data/jibenmian/{}-{}-{}.csv'.format(_YEJI_, year, season)
            logger.info("Starting thread: {}, to data: {}".format(self.name, file_name))
            ts.get_report_data(year, season).to_csv(file_name)
            logger.info('Successfully fetched jbm data to {}'.format(file_name))

            file_name = 'data/jibenmian/{}-{}-{}.csv'.format(_PROFIT_, year, season)
            logger.info("Starting thread: {}, to data: {}".format(self.name, file_name))
            ts.get_profit_data(year, season).to_csv(file_name)
            logger.info('Successfully fetched jbm data to {}'.format(file_name))

            file_name = 'data/jibenmian/{}-{}-{}.csv'.format(_OPERATION_, year, season)
            logger.info("Starting thread: {}, to data: {}".format(self.name, file_name))
            ts.get_operation_data(year, season).to_csv(file_name)
            logger.info('Successfully fetched jbm data to {}'.format(file_name))

            file_name = 'data/jibenmian/{}-{}-{}.csv'.format(_GROWTH_, year, season)
            logger.info("Starting thread: {}, to data: {}".format(self.name, file_name))
            ts.get_growth_data(year, season).to_csv(file_name)
            logger.info('Successfully fetched jbm data to {}'.format(file_name))

            file_name = 'data/jibenmian/{}-{}-{}.csv'.format(_DEBTPAYING_, year, season)
            logger.info("Starting thread: {}, to data: {}".format(self.name, file_name))
            ts.get_debtpaying_data(year, season).to_csv(file_name)
            logger.info('Successfully fetched jbm data to {}'.format(file_name))

            file_name = 'data/jibenmian/{}-{}-{}.csv'.format(_CASHFLOW_, year, season)
            logger.info("Starting thread: {}, to data: {}".format(self.name, file_name))
            ts.get_cashflow_data(year, season).to_csv(file_name)
            logger.info('Successfully fetched jbm data to {}'.format(file_name))

        except Exception as e:
            print('Error occured: {}'.format(e))

# 创建新线程
#threads.append(myThread(_BASICS_))
year = 2015
season = 1
i = 0
for year in [2016,2017]:
    for season in [1, 2,3,4]:
        threads.append(myThread(i, year, season))
        i = i + 1
    i = i + 1
threads.append(myThread(i+1, 2018, 1))

# 开启新线程
for thread in threads:
    thread.start()

# 等待所有线程完成
for t in threads:
    t.join()
