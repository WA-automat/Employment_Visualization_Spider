from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json
from selenium.webdriver import ActionChains

# 浏览器对象
browser = webdriver.Edge(executable_path=r'.\msedgedriver.exe')

# 创建 ActionChains 对象：类似一个鼠标
action = ActionChains(browser)

if __name__ == '__main__':
    # 获取职业列表
    browser.get("https://www.kanzhun.com/")
    job_select = browser.find_element(By.XPATH, "//div[@class='select-job']")
    job_select.click()
    time.sleep(1)
    job_dict = {}
    list_blocks = browser.find_elements(By.CLASS_NAME, "list-block")
    for list_block in list_blocks:
        name = list_block.find_element(By.CLASS_NAME, "second-name").text
        job_dict[name] = []
        jobs = list_block.find_elements(By.CLASS_NAME, "cell-item")
        for job_item in jobs:
            job_dict[name].append(job_item.text)
    categories = browser.find_elements(By.CLASS_NAME, "list-item")
    for category in categories:
        action.move_to_element(category).perform()
        list_blocks = browser.find_elements(By.CLASS_NAME, "list-block")
        for list_block in list_blocks:
            name = list_block.find_element(By.CLASS_NAME, "second-name").text
            job_dict[name] = []
            jobs = list_block.find_elements(By.CLASS_NAME, "cell-item")
            for job_item in jobs:
                job_dict[name].append(job_item.text)

    print(job_dict)
    # for job in job_list:
    #     browser.get(f"https://www.kanzhun.com/search/?query={job}&type=0")
    #     time.sleep(1)
    data_dict = {}
    for cate, jobs in job_dict.items():
        data_dict[cate] = {}
        for job in jobs:
            try:
                data_dict[cate][job] = {}
                browser.get(f"https://www.kanzhun.com/search/?query={job}&type=0")
                time.sleep(2)
                numbers = browser.find_elements(By.CLASS_NAME, "red")
                data_dict[cate][job]['平均数'] = numbers[0].text
                data_dict[cate][job]['中位数'] = numbers[1].text
                search_datas = browser.find_elements(By.CLASS_NAME, "search-data-item")
                data_dict[cate][job]['就业趋势'] = search_datas[0].text
                data_dict[cate][job]['招聘趋势'] = search_datas[1].text
            except Exception:
                continue
    browser.quit()

    with open('../data/job_data.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)
