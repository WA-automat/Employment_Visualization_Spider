# -*- coding:utf-8 -*-
# @File :数据特征工程1.py
# @Software: PyCharm

import pandas as pd
import numpy as np


from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch #此处主要用来画图

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#数据预处理
def data_pretreatment(data_df0):
    data_df1 = data_df0.copy()

    # 将revenue列由文本型数据转为数值型数据
    for i in range(len(data_df0)):

        r_t=data_df0.loc[i,'revenue'].strip()
        r_t_list = r_t.split('-')
        r_bottom = float(r_t_list[0][:-1]) * 1000  # 下限
        r_top = float(r_t_list[-1][:-1]) * 1000  # 上限

        data_df1.loc[i, 'revenue'] = (r_bottom + r_top) / 2  # 平均薪资



    # print(data_df1.loc[:,'revenue'])


    # 对experience,attribute,nop,qualification 列进行标签编码
    # 且因为后续需要对成功应聘的难度进行评价，故标签编码的数值越大，代表难度越大

    label_code_dict={
        #工作经验
        '经验不限':0,'在校/应届':0,'1年以下':1,'1-3年':2,'3-5年':3,'5-10年':4,'10年以上':5,
        #工作性质
        '实习':0,'兼职':1,'全职':2,
        #团队规模
        '<15人':0,'15-50人':1,'50-150人':2,'150-500人':3,'500-2000人':4,'2000人>':5,
        #学历
        '学历不限':0,'大专':1,'本科':2,'硕士':3,'博士':4
    }

    label_list=['experience','attribute','nop','qualification']
    for i in range(len(data_df0)):
        for lb in label_list:
            temp_t=data_df0.loc[i,lb].strip()
            data_df1.loc[i, lb] =label_code_dict[temp_t]

    # print(data_df1.loc[:,label_list])

    # 预处理后表格
    writer1=pd.ExcelWriter('招聘信息_afterProcessing.xlsx')
    data_df1.to_excel(writer1,sheet_name='sheet_1',index=False)
    writer1.save()#记得加save

    return data_df1


# 以下进行如下几项特征挖掘：
# 一、分析每个城市的平均薪资与('experience','attribute','nop','qualification')的分布
# 二、将招聘数据以('revenue','experience','attribute','qualification')等作为评价指标，
#    进行以变异系数为权重的TOPSIS模型评价后，得到每个招聘工作的应聘成功难度评分



# 一、
def feature_analyse1(fa1_df0,fa1_df1):
    # 每个城市平均薪资

    fa1_df2=fa1_df1.groupby(['site',]).agg({'revenue':['count','sum','mean' ]})
    mean_r_s=fa1_df2.loc[:,('revenue','mean')]
    # print(mean_r_s)
    city_name_list=mean_r_s.index.tolist()
    print(city_name_list,len(city_name_list))
    mean_r_list=mean_r_s.values.round(2).tolist()
    # print(mean_r_list)


    # 每个城市('experience','attribute','nop','qualification')的分布

    labels_distribution={
        'experience':[], #每行为一个城市的分布，每列含义依次为labels_explanations[0]
        'attribute':[],#每行为一个城市的分布，每列含义依次为labels_explanations[1]
        'nop':[],#每行为一个城市的分布，每列含义依次为labels_explanations[2]
        'qualification':[] #每行为一个城市的分布，每列含义依次为labels_explanations[3]
    }
    labels_explanations= {
    'experience':['经验不限','在校/应届' ,'1年以下' ,'1-3年' ,'3-5年' ,'5-10年' ,'10年以上' ],
    'attribute':['兼职' ,'全职' ,'实习' ,'#','#','#','#'],
    'nop':['<15人' ,'15-50人' ,'50-150人' ,'150-500人' ,'500-2000人' ,'2000人>','#'],
    'qualification':['学历不限' ,'大专','本科' ,'硕士' ,'博士' ,'#','#']
    }

    for label in ['experience','attribute','nop','qualification']:
        one_explanation=labels_explanations[label]
        fa1_t = fa1_df0.groupby(['site', label]).agg({label: ['count']})
        # print(fa1_t)

        for one_city in city_name_list:
            distribution_t_list=[] #某一城市(取决于one_city)的某一分布(取决于label)
            for meaning in one_explanation:
                try:
                    distribution_t_list.append(fa1_t.loc[(one_city, meaning),(label,'count')])
                except:
                    distribution_t_list.append(0)
            labels_distribution[label].append(distribution_t_list)

    # print(labels_distribution)
    return city_name_list,mean_r_list,labels_explanations,labels_distribution

#二
def feature_analyse2(fa2_df0):
    evaluate_data=fa2_df0.loc[:,['revenue','experience','attribute','nop','qualification']].values
    evaluate_data=evaluate_data.astype('float')

    temp_num= np.sqrt((evaluate_data**2).sum(axis=0))
    evaluate_data=evaluate_data/temp_num #标准化
    max_num=evaluate_data.max(axis=0)
    min_num=evaluate_data.min(axis=0)

    # 结合变异系数法的TOPSIS评价模型
    W= np.std(evaluate_data,ddof=1,axis=0) /np.mean(evaluate_data,axis=0)#变异系数法权重
    W= W/W.sum()
    print(W)

    D_best=  np.sqrt((  ((evaluate_data- max_num)**2)*W  ).sum(axis=1))
    D_worst = np.sqrt(( ((evaluate_data - min_num) ** 2)*W  ).sum(axis=1))
    D_sum=D_best+D_worst
    C=D_worst/D_sum #综合评分
    print((C>0.5).sum(),C.shape)

    fa2_df0.loc[:,'degree of difficulty']=C #新增一列难度评分

    #
    writer1=pd.ExcelWriter('招聘信息_afterProcessing.xlsx')
    fa2_df0.to_excel(writer1,sheet_name='sheet_1',index=False)
    writer1.save()#记得加save

    return fa2_df0.loc[:,['site','degree of difficulty']]


Data_df0 = pd.read_excel('招聘信息汇总.xlsx', sheet_name=0)
# # print(Data_df0)
# Data_df1=data_pretreatment(Data_df0)


# feature_analyse1(Data_df0,Data_df1)
# feature_analyse2(Data_df1)



