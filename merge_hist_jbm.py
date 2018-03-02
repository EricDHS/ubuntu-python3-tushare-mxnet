# -*- coding: utf-8 -*-
import urllib
import re
import math
import sys
import os
import threading
import datetime
from datetime import datetime
import tushare as ts
import threading
import time
import logging
import subprocess
import pandas as pd
import numpy as np
import traceback

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
_BASICS_ = 'basics_report'
_YEJI_ = 'yeji_report'
_PROFIT_ = 'profit_report'
_OPERATION_= 'operation_report'
_GROWTH_ = 'grown_report'
_DEBTPAYING_ = 'debat_paying_report'
_CASHFLOW_ = 'cash_flow_report'

class myThread (threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
    def run(self):
        while True:
            code = get_code()
            if not code:
                return
            logger.info("Starting name: {}, merging data for code: {}".format(self.name, code))
            create_data_mxnet(code)
def create_data_mxnet(code):
    try:
        hist_data = pd.read_csv('data/hist/{}_hist.csv'.format(code)).iloc[::-1]
        hist_data.reset_index(drop=True, inplace=True)
        yeji_data = pd.read_csv('data/jibenmian/all_yeji_report.csv').sort_values('report_date')
        yeji_data = yeji_data.loc[yeji_data['code']==int(code)]
        hist_data['date'] =  pd.to_datetime(hist_data['date'], format='%Y-%m-%d')
        hist_data['yeji_eps'] = np.NaN
        hist_data['yeji_eps_yoy'] = np.NaN
        hist_data['yeji_bvps'] = np.NaN
        hist_data['yeji_roe'] = np.NaN
        hist_data['yeji_epcf'] = np.NaN
        hist_data['yeji_net_profits'] = np.NaN
        hist_data['yeji_profits_yoy'] = np.NaN
        hist_data['yeji_distrib'] = np.NaN
        begin_date = None
        logger.info('Begin to Ana Yeji data')
        for index,row in yeji_data.iterrows():
            date = datetime.strptime(row['report_date'], '%Y-%m-%d')
            if begin_date:
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_eps'] = row['eps']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_eps_yoy'] = row['eps_yoy']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_bvps'] = row['bvps']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_roe'] = row['roe']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_epcf'] = row['epcf']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_net_profits'] = row['net_profits']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_profits_yoy'] = row['profits_yoy']
                hist_data.loc[(hist_data['date']>=begin_date) & (hist_data['date']<date), 'yeji_distrib'] = row['distrib']
                begin_date = date
            else:
                hist_data.loc[hist_data['date']<date, 'yeji_eps'] = row['eps']
                hist_data.loc[hist_data['date']<date, 'yeji_eps_yoy'] = row['eps_yoy']
                hist_data.loc[hist_data['date']<date, 'yeji_bvps'] = row['bvps']
                hist_data.loc[hist_data['date']<date, 'yeji_roe'] = row['roe']
                hist_data.loc[hist_data['date']<date, 'yeji_epcf'] = row['epcf']
                hist_data.loc[hist_data['date']<date, 'yeji_net_profits'] = row['net_profits']
                hist_data.loc[hist_data['date']<date, 'yeji_profits_yoy'] = row['profits_yoy']
                hist_data.loc[hist_data['date']<date, 'yeji_distrib'] = row['distrib']
                begin_date = date
        #hist_data.set_index(drop=True)
        care_types = ['high', 'low', 'open', 'close']
        for t in care_types:
            c_data = hist_data[t]
            result = hist_data
            new_column = 'predict_{}'.format(t)
            result[new_column] = np.NaN
            result.reset_index(drop=True, inplace=True)
            c_data.drop(c_data.head(1).index, inplace=True)
            c_data.reset_index(drop=True, inplace=True)
            result[new_column] = c_data
            result.drop(result.tail(1).index,inplace=True)
            result.to_csv('data/mxnet/{}-{}.csv'.format(t, code))

    except Exception as e:
        logger.error('Error occured: {}'.format(e))
        traceback.print_exc()

# 创建新线程
#threads.append(myThread(_BASICS_))
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
