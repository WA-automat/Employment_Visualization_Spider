# -*- coding:utf-8 -*-
# @File :岗位招聘信息爬取.py
# @Software: PyCharm
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver import ActionChains

browser = webdriver.Edge(executable_path=r'.\msedgedriver.exe')  # 实例化一个驱动对象

browser.get('https://www.lagou.com')
time.sleep(1.2)
browser.find_element(By.LINK_TEXT, '登录').click()
time.sleep(1)
browser.find_element(By.XPATH,
                     '/html/body/div[12]/div/div[2]/div/div[2]/div/div[2]/div[3]/div[1]/div/div[2]/div[3]/div').click()  # 密码登录
browser.find_element(By.NAME, 'account').send_keys('13613083494')
browser.find_element(By.NAME, 'password').send_keys('Lxh@13613083494')
browser.find_element(By.XPATH,
                     '/html/body/div[12]/div/div[2]/div/div[2]/div/div[2]/div[3]/div[3]/div[2]/div[2]/div').click()  # 勾选同意

browser.find_element(By.XPATH,
                     '/html/body/div[12]/div/div[2]/div/div[2]/div/div[2]/div[3]/div[2]/button').click()  # 点登录
time.sleep(6)

time.sleep(4)
browser.find_element(By.XPATH, '/html/body/div[6]/div[1]/div/div[1]/form/input[5]').click()  # 点搜索

data = browser.page_source  # 获取页面的源代码
# print(data)
#

#
# '哈尔滨' ,'合肥' ,'福州' ,'南昌' ,'济南' ,
# '兰州'  ,'银川', '乌鲁木齐',
# '海口','贵阳', '重庆' , '石家庄' ,'昆明','郑州',
# '北京','上海','深圳', '广州','杭州', '成都', '武汉',  '西安',  '长沙', '苏州' ,'天津','厦门', '南京',

# 过少： '拉萨' ,'西宁'
city_list = ['呼和浩特', '沈阳', '长春', '南宁', '太原', ]  #

# 自动添加所需城市
# city_test=browser.find_elements(By.XPATH,'//*[@id="jobsContainer"]/div[2]/div[1]/div[1]/div[1]/div[1]/div/div[2]/*')
# time.sleep(1)
# for i in range(len(city_test)):
#     city_test = browser.find_elements(By.XPATH,'//*[@id="jobsContainer"]/div[2]/div[1]/div[1]/div[1]/div[1]/div/div[2]/*')#此处必须要重新定位一次，不然会报web元素不稳定的错误
#     city_list.append(city_test[i].text)
# print(city_list[1:])
# city_list=city_list[1:]
# time.sleep(0.5)


for city_name in city_list:
    time.sleep(2)
    city_total = dict()
    for pn in [1, 2, 3]:  # [i for i in range(1,5)]
        print(pn)
        temp_url = 'https://www.lagou.com/wn/jobs?fromSearch=true&pn=' + str(pn) + '&city=' + city_name
        browser.get(temp_url)
        action = ActionChains(browser)
        job_elements = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')

        for j in range(len(job_elements)):
            company_dict = dict()
            time.sleep(1)

            try:
                one_job = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')[
                    j]  # 此处必须要重新定位一次，不然会报web元素不稳定的错误
                time.sleep(0.5)
                company = one_job.find_element(By.XPATH, './/div[1]/div[2]/div[1]/a').text
                company_dict['com_name'] = company
            except:
                company_dict['com_name'] = ''

            try:
                one_job = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')[
                    j]  # 此处必须要重新定位一次，不然会报web元素不稳定的错误
                time.sleep(0.5)
                inf = one_job.find_element(By.XPATH, './/div[1]/div[2]/div[2]').text
                inf_list = inf.split('/')
                company_dict['type'] = inf_list[0].strip()
                company_dict['nop'] = inf_list[-1].strip()
            except:
                company_dict['type'] = ''
                company_dict['nop'] = ''

            try:
                one_job = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')[
                    j]  # 此处必须要重新定位一次，不然会报web元素不稳定的错误
                time.sleep(0.5)
                feature = one_job.find_element(By.XPATH, './/div[2]/div[@class="il__3lk85"]').text
                company_dict['feature'] = feature
            except:
                company_dict['feature'] = ''

            one_job = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')[
                j]  # 此处必须要重新定位一次，不然会报web元素不稳定的错误
            time.sleep(0.8)
            temp_a = one_job.find_element(By.XPATH, './/div[1]/div[1]/div[1]/a')
            # print(temp_a.text)
            try:
                action.move_to_element(temp_a).perform()  # 鼠标悬停
                time.sleep(1)  #
                print('j', j)
                floating_box = browser.find_element(By.XPATH, '/html/body/div[' + str(j + 2) + ']')

                temp_details = floating_box.find_element(By.XPATH, './/div/div/div/div/div').text
                time.sleep(0.8)
                detail_list = temp_details.split('\n')

                temp_title = ['job', 'revenue', 'site', 'experience', 'qualification', 'attribute']
                for index in range(len(temp_title)):
                    company_dict[temp_title[index]] = ''
                for index in range(len(detail_list)):
                    company_dict[temp_title[index]] = detail_list[index].strip()

                floating_box = browser.find_element(By.XPATH,
                                                    '/html/body/div[' + str(j + 2) + ']')  # 此处必须要重新定位一次，不然会报web元素不稳定的错误
                detailed_requirement = floating_box.find_element(By.XPATH, './/div/div/div/div[2]/div[2]/div').text
                company_dict['detail'] = detailed_requirement

                #
                t_key = company_dict['com_name'] + ':' + company_dict['job']
                city_total[t_key] = company_dict
                print(t_key)
                print(company_dict)
                print('********************')
                time.sleep(0.8)
            except:
                continue
    # 增量式
    try:
        pre_f = open(city_name + '_total.json', 'r', encoding='utf-8')
        pre_d = json.load(pre_f)  # 用.load
        pre_f.close()

        with open(city_name + '_total.json', 'w', encoding='utf-8') as fp:
            pre_d.update(city_total)
            json.dump(pre_d, fp, indent=4, ensure_ascii=False)
            # 此时也要写参数ensure_ascii=False，不然json文件中还是不显示中文，显示ascii值
    except:
        with open(city_name + '_total.json', 'w', encoding='utf-8') as fp:
            json.dump(city_total, fp, indent=4, ensure_ascii=False)
    time.sleep(1)
