import glob,os
import pandas as pd
path = 'data/jibenmian/'
os.chdir(path)
all_data = pd.DataFrame()
for f in glob.glob("yeji_report-*"):
    data = pd.read_csv('{}'.format(f))
    name = os.path.splitext(f)[0]
    year = int(name.split('-')[1])
    season = int(name.split('-')[2])
    full_report_date = []
    for date in data['report_date']:
        month = int(date.split('-')[0])
        if month <= (season*3):
           new_date = '{}-{}'.format(year+1, date)
        else:
           new_date = '{}-{}'.format(year, date)
        full_report_date.append(new_date)
    data['report_date']  = full_report_date
    all_data = all_data.append(data, ignore_index=True)

all_data.to_csv('all_yeji_report.csv')    
    
