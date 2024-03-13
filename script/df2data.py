import json

import jieba
import pandas as pd
from collections import Counter

if __name__ == '__main__':
    with open('../data/stopwords.txt', "r", encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [line.rstrip('\n') for line in stopwords]

    # 表格转json
    df = pd.read_excel("../data/招聘信息汇总.xlsx", sheet_name='sheet_1')
    df_afterProcessing = pd.read_excel("../data/招聘信息_afterProcessing.xlsx", sheet_name='sheet_1')
    df.to_json("../data/招聘信息汇总.json", orient='records', force_ascii=False)
    df_afterProcessing.to_json("../data/招聘信息_afterProcessing.json", orient='records', force_ascii=False)

    # 获取不同城市
    citys = list(set(df['site'].values.tolist()))

    cloud_data = {}
    # 转换词云图数据
    for city in citys:
        city_df = df[df['site'] == city]
        # 公司名字、职业、特点
        com_name = city_df['com_name'].values.tolist()
        job = [word for s in city_df['job'].values.tolist() for word in jieba.lcut(s) if word not in stopwords]
        feature = [word for s in city_df['feature'].values.tolist() for word in jieba.lcut(str(s)) if word not in stopwords]

        com_name_counter = Counter(com_name)
        job_counter = Counter(job)
        feature_counter = Counter(feature)

        cloud_data[city] = {}
        cloud_data[city]['com_name'] = []
        for key, value in com_name_counter.items():
            cloud_data[city]['com_name'].append({'name': key, 'value': value})

    output_file = "../data/word_counts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cloud_data, f, ensure_ascii=False, indent=4)
