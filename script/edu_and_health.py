import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
import urllib.request
import re
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54'
}

name_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32]

provinces = [
    '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆', '台湾',
    '香港', '澳门'
]


def get_page_data(data_url, header):
    req = urllib.request.Request(data_url, headers=header)
    content = urllib.request.urlopen(req).read()  # .decode('GBK')
    content = content.decode('utf-8')  # python3
    page = BeautifulSoup(content, 'html.parser')
    return page


def analyse_data(page, year):
    table = page.find('table', attrs={'id': 'edu_login'})
    trs = table.find_all('tr')[2:]
    df_data = pd.DataFrame(columns=['date', 'price'])
    count = 0

    for tr in trs:
        tds = tr.find_all('td')
        date = tds[0].text
        # date = get_date(date, year)
        new = tds[2].text
        new = new[:4]
        # print(order+":"+date+"***"+old+"***"+new)
        df_data.loc[count] = [date, new]
        count += 1
    # df_data.set_index('date',inplace=True)
    return df_data


if __name__ == '__main__':
    json_data = {'教育': {}, '卫生': {}}
    # 爬取教育行业数据
    for i in name_list:
        try:
            url = f"https://www.gotohui.com/edu/edata-{i}"
            page = get_page_data(url, headers)
            name_a = page.find('a', attrs={'class': 'name', 'href': f'https://www.gotohui.com/area/{i}'})
            if name_a is not None:
                name = name_a.get_text()
                table = page.find('table', attrs={'id': 'edu_login'})
                data_2017 = table.find('tr', {'index': '2017'})

                # 找到2017年大学生数量
                data_second_cell_2017 = data_2017.find_all('td')[2].text
                json_data['教育'][name] = data_second_cell_2017

            url = f"https://www.gotohui.com/health/hdata-{i}"
            page = get_page_data(url, headers)
            name_a = page.find('a', attrs={'class': 'name', 'href': f'https://www.gotohui.com/area/{i}'})
            if name_a is not None:
                name = name_a.get_text()
                table = page.find('table', attrs={'id': 'health_login'})
                data_2017 = table.find('tr', {'index': '2017'})

                # 找到2017年大学生数量
                data_second_cell_2017 = data_2017.find_all('td')[1].text
                json_data['卫生'][name] = data_second_cell_2017
        except Exception:
            continue

    with open('../data/edu_and_health.json', 'w', encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
