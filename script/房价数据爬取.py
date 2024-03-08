# -*- coding:utf-8 -*-
# @File :房价数据爬取.py
# @Software: PyCharm


from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json
from selenium.webdriver import ActionChains

browser = webdriver.Edge(executable_path=r'.\msedgedriver.exe')
# browser = webdriver.Firefox(executable_path=r'D:\python\python安装\Scripts\geckodriver.exe')#实例化一个驱动对象
city_list = ['北京', '上海', '深圳', '广州', '杭州', '成都', '南京', '武汉', '西安', '厦门', '长沙', '苏州', '天津',
             '石家庄', '太原', '呼和浩特', '沈阳', '长春',
             '哈尔滨', '合肥', '福州', '南昌', '济南',
             '郑州', '南宁', '海口', '重庆', '贵阳',
             '昆明', '拉萨', '兰州', '西宁', '银川', '乌鲁木齐']

city_abbr = {
    '北京': 'bj', '上海': 'sh', '深圳': 'sz', '广州': 'gz', '杭州': 'hz',
    '成都': 'cd', '南京': 'nj', '武汉': 'wh', '西安': 'xa', '厦门': 'xm',
    '长沙': 'cs', '苏州': 'suz', '天津': 'tj',
    '石家庄': 'sjz', '太原': 'ty', '呼和浩特': 'huhehaote', '沈阳': 'sy', '长春': 'changchun',
    '哈尔滨': 'haerbin', '合肥': 'hefei', '福州': 'fz', '南昌': 'nanchang', '济南': 'jn',
    '郑州': 'zz', '南宁': 'nanning', '海口': 'haikou', '重庆': 'cq', '贵阳': 'guiyang',
    '昆明': 'kunming', '拉萨': 'lasa', '兰州': 'lanzhou', '西宁': 'xining', '银川': 'yinchuan', '乌鲁木齐': 'xj'
}

fj_data_sum = {}

if __name__ == '__main__':

    for city_name in city_list:
        url_template = 'http://{}.fangjia.com/trend/year2Data?defaultCityName={}&districtName=&region=&block=&keyword='.format(
            city_abbr[city_name], city_name)

        browser.get(url_template)
        time.sleep(1)
        print(city_name, ":")
        fj_t = browser.find_element(By.XPATH, '/html/body').text
        fj_t = json.loads(fj_t)
        # print(fj_t, type(fj_t))

        detailed_data = fj_t['series'][0]['data']
        print(detailed_data, type(detailed_data))
        fj_data_sum[city_name] = detailed_data

    with open('房价' + '_total.json', 'w', encoding='utf-8') as fp:
        json.dump(fj_data_sum, fp, indent=4, ensure_ascii=False)
