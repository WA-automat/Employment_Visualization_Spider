# -*- coding:utf-8 -*-
# @File :存入MongoDB.py
# @Software: PyCharm

import json

from pymongo import MongoClient

# 连接数据库
client = MongoClient("mongodb://localhost:27017/")
# Database Name
db = client['招聘信息']

#对原始json数据进行预处理，剔除无用数据
city_list=['北京', '上海', '深圳','广州','杭州', '成都',  '南京','武汉','西安', '厦门','长沙','苏州','天津',
           '石家庄' , '太原' ,'呼和浩特' ,'沈阳', '长春',
           '哈尔滨' ,'合肥' ,'福州' ,'南昌' ,'济南' ,
           '郑州', '南宁', '海口','重庆' ,'贵阳',
           '昆明', '拉萨'  ,'兰州' ,'西宁' ,'银川', '乌鲁木齐']
print(len(city_list))

for city_name in city_list:
    print(city_name)
    pre_f = open(city_name + '_total.json', 'r', encoding='utf-8')
    pre_d = json.load(pre_f)  # 用.load
    pre_f.close()

    temp_items1=list(pre_d.items())
    for k1,v1 in temp_items1:
        judge1=0
        for t_split in k1.split(':'):
            if  t_split.strip()=='':
                pre_d.pop(k1)
                judge1=1
                break
        if judge1==0:
            temp_items2 = list(v1.items())
            for k2,v2 in temp_items2:
                if v2=='':
                    pre_d.pop(k1)
                    break
                elif k2=='com_name' and v2[0]=='某':
                    pre_d.pop(k1)
                    break

    temp_items1 = list(pre_d.items())
    for k1, v1 in temp_items1:
        for k2, v2 in v1.items():
            if k2=='experience' and v2=='不限':
                pre_d[k1][k2]='经验不限'
            elif k2=='qualification' and v2=='不限':
                pre_d[k1][k2] = '学历不限'
            elif k2 == 'nop' and v2 == '少于15人':
                pre_d[k1][k2] = '<15人'
            elif k2 == 'nop' and v2 == '2000人以上':
                pre_d[k1][k2] = '2000人>'

    with open(city_name + '_total.json', 'w', encoding='utf-8')as fp:
        json.dump(pre_d, fp, indent=4, ensure_ascii=False)
        # 此时也要写参数ensure_ascii=False，不然json文件中还是不显示中文，显示ascii值

    #存入MongoDB数据库
    col = db[city_name]
    col.drop()

    for job in pre_d.keys():
        col.insert_one(pre_d[job])


db2 = client['房价信息']

fj_sum = open( '房价_total.json', 'r', encoding='utf-8')
fj_sum_data = json.load(fj_sum)  # 用.load
fj_sum.close()

col = db2['近两年每周平均房价信息']
col.drop()

col.insert_one(fj_sum_data )
