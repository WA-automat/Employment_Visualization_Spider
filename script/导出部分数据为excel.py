# -*- coding:utf-8 -*-
# @File :导出部分数据为excel.py
# @Software: PyCharm


#导出为excel
import pandas as pd
import numpy as np
from pymongo import MongoClient


# 连接数据库
client = MongoClient("mongodb://localhost:27017/")
# Database Name
db = client['招聘信息']

data_df=pd.DataFrame(columns=['id','job','com_name','site','revenue','experience','attribute','type','nop','feature'])

city_list=['北京', '上海', '深圳','广州','杭州', '成都',  '南京','武汉','西安', '厦门','长沙','苏州','天津',
           '石家庄' , '太原' ,'呼和浩特' ,'沈阳', '长春',
            '哈尔滨' ,'合肥' ,'福州' ,'南昌' ,'济南' ,
           '郑州', '南宁', '海口','重庆' ,'贵阳',
           '昆明', '拉萨'  ,'兰州' ,'西宁' ,'银川', '乌鲁木齐']
index=0
for city_name in city_list:

    col = db[city_name]
    jobs_collection=col.find()

    symbol_cleaning=[" ","\"" ,"“","”","*","!","！","。","."]
    for one_job in jobs_collection:
        for k1,v1 in one_job.items():
            temp_v=v1
            if k1!='detail' and k1!='_id':
                if k1 != 'type' or k1 != 'feature':
                    for symbol in symbol_cleaning:
                        temp_v=temp_v.strip(symbol)

                data_df.loc[index,k1]=temp_v
        data_df.loc[index, 'id'] =index
        index+=1


writer1=pd.ExcelWriter('招聘信息汇总.xlsx')
data_df.to_excel(writer1,sheet_name='sheet_1',index=False)

writer1.save()#记得加save