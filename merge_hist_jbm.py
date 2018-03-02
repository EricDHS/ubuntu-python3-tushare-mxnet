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
import pandas as pd
import numpy as np

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
            yeji_data = pd.read_csv('data/jibenmian/all_yeji_report.csv').sort('date_report')
            yeji_data = yeji_data.loc[yeji_data['code']==int(self.code)]
            
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
                if begin_date:
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_eps'] = row['eps']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_eps_yoy'] = row['eps_yoy']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_bvps'] = row['bvps']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_roe'] = row['roe']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_epcf'] = row['epcf']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_net_profits'] = row['net_profits']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_profits_yoy'] = row['profits_yoy']
                    hist_data.loc[hist_data['date']>=begin_date and hist_data['date']<row['report_date']]['yeji_distrib'] = row['distrib']
                    begin_date = row['report_date']
                else:
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_eps'] = row['eps']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_eps_yoy'] = row['eps_yoy']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_bvps'] = row['bvps']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_roe'] = row['roe']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_epcf'] = row['epcf']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_net_profits'] = row['net_profits']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_profits_yoy'] = row['profits_yoy']
                    hist_data.loc[hist_data['date']<row['report_date']]['yeji_distrib'] = row['distrib']
                    begin_date = row['report_date']

        except Exception as e:
            logger.error('Error occured: {}'.format(e))

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
