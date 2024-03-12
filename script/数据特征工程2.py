# -*- coding:utf-8 -*-
# @File :数据特征工程2.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import json
import copy

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold, cross_val_score  # 交叉验证
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 超参数搜索
from imblearn.under_sampling import RandomUnderSampler

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch  # 此处主要用来画图
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller  # 平稳性检验
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 自相关图、偏自相关图


# 特征挖掘_续：
# 三、先对revenue变量进行系统聚类，此处得薪资水平分为6类，然后统计每个城市的薪资水平分布
# 四、分析所有招聘信息中，'com_name','type','feature' 等列中所出现关键词的词频，即哪些关键词热度高
# 五、基于各城市近两年平均每周的房价，使用ARIMA时间序列模型，对其进行后5周的平均房价的预测。


# 三
def feature_analyse3():
    file_data = pd.read_excel('../data/招聘信息_afterProcessing.xlsx')
    file_data = file_data.loc[:, ['id', 'site', 'revenue', 'experience', 'attribute', 'nop', 'qualification']]

    new_arr1 = file_data.loc[:, 'revenue'].values.reshape(-1, 1)

    # 画层次聚类图
    # z=sch.linkage(new_arr1,method='ward')
    # sch.dendrogram(z ,new_arr1.shape[0])
    # plt.title('Cluster')
    # plt.show()

    # S=[] #存总轮廓系数 （公式看word）
    # K=range(2,8) #K取值从2到样本容量n-1，不能取n因为取n时总轮廓系数为1，且无意义
    # for i in K:
    #     Agg_hc = AgglomerativeClustering(n_clusters=i, metric='euclidean',
    #                                      linkage='ward', )  # linkage = 'ward'时，metric必须为'euclidean'，即默认
    #     labels = Agg_hc.fit_predict(new_arr1)  # 训练数据
    #     S.append(silhouette_score(new_arr1,labels))
    #
    # plt.plot(K,S,'o-')
    # plt.show()

    Agg_hc = AgglomerativeClustering(n_clusters=6, metric='euclidean',
                                     linkage='ward', )  # linkage = 'ward'时，metric必须为'euclidean'，即默认
    y_hc = Agg_hc.fit_predict(new_arr1)  # 训练数据
    # 也可以使用k-means方法，此处用系统聚类法
    # md=KMeans(6,n_init=10).fit(new_arr1)
    # y_hc=md.labels_
    # print(y_hc)#查看每个样本属于哪个族
    file_data.loc[:, 'revenue_class'] = y_hc  # 分为6类

    file_data1 = file_data.loc[:, ['revenue', 'revenue_class']]
    file_data1 = file_data1.groupby(['revenue_class']).agg({'revenue': ['mean', 'count']})
    print(file_data1)
    class_explanation = {}
    for i in range(len(file_data1)):
        class_explanation[str(i)] = file_data1.loc[i, ('revenue', 'mean')].round(2)
    print(class_explanation)
    # ------------

    # #第二次
    # new_file_data=file_data.loc[( file_data.loc[:,'revenue_class']==1),:]
    # # print(new_file_data)
    # new_arr2=new_file_data.loc[:,'revenue'].values.reshape(-1,1)
    #
    # # 画层次聚类图
    # # z=sch.linkage(new_arr2,method='average')
    # # sch.dendrogram(z ,new_arr2.shape[0])
    # # plt.title('Cluster')
    # # plt.show()
    #
    # # S=[] #存总轮廓系数 （公式看word）
    # # K=range(2,7) #K取值从2到样本容量n-1，不能取n因为取n时总轮廓系数为1，且无意义
    # # for i in K:
    # #     Agg_hc = AgglomerativeClustering(n_clusters=i, metric='euclidean',
    # #                                      linkage='ward', )  # linkage = 'ward'时，metric必须为'euclidean'，即默认
    # #     labels = Agg_hc.fit_predict(new_arr2)  # 训练数据
    # #     S.append(silhouette_score(new_arr2,labels))
    # #
    # # plt.plot(K,S,'o-')
    # # plt.show()
    #
    # Agg_hc = AgglomerativeClustering(n_clusters =2, metric = 'euclidean', linkage = 'average',) #linkage = 'ward'时，metric必须为'euclidean'，即默认
    # y_hc = Agg_hc.fit_predict(new_arr2) # 训练数据
    # #也可以使用k-means方法，此处用系统聚类法
    # # md=KMeans(5,n_init=10).fit(new_arr2)
    # # y_hc=md.labels_
    # # print(y_hc)#查看每个样本属于哪个族
    # new_file_data.loc[:,'revenue_class']=y_hc #分为4类
    #
    # file_data2=new_file_data.loc[:,['revenue','revenue_class']]
    # file_data2= file_data2.groupby(['revenue_class']).agg({'revenue': ['mean','count']})
    # print(file_data2)
    #
    # class_explanation={}
    # for i in range(len(file_data2)):
    #     class_explanation[str(i)]=file_data2.loc[i,('revenue','mean')].round(2)
    # print(class_explanation)
    # #----------
    #
    #
    # #该原表，整合类别
    # for i in range(len(file_data)):
    #     if file_data.loc[i,'revenue_class']==1:
    #         file_data.loc[i, 'revenue_class'] = new_file_data.loc[i, 'revenue_class']
    #     else:
    #         if file_data.loc[i, 'revenue_class']==0:
    #             file_data.loc[i, 'revenue_class']=2
    #         elif file_data.loc[i, 'revenue_class']==2:
    #             file_data.loc[i, 'revenue_class']=3
    #         elif file_data.loc[i, 'revenue_class']==3:
    #             file_data.loc[i, 'revenue_class']=4
    #         else:
    #             file_data.loc[i, 'revenue_class'] = 5
    #
    # file_data1=file_data.loc[:,['revenue','revenue_class']]
    # file_data1= file_data1.groupby(['revenue_class']).agg({'revenue': ['mean','count']})
    # print(file_data1)

    class_explanation = {}
    for i in range(len(file_data1)):
        class_explanation[str(i)] = file_data1.loc[i, ('revenue', 'mean')].round(2)
    class_explanation = [(i, j) for i, j in class_explanation.items()]
    class_explanation = sorted(class_explanation, key=lambda x: x[1])
    print(class_explanation)

    writer1 = pd.ExcelWriter('分类结果测试.xlsx')
    file_data.to_excel(writer1, sheet_name='sheet_1', index=False)
    writer1.save()  # 记得加save

    return file_data, class_explanation


# 四
def feature_analyse4(city_list):
    file_data = pd.read_excel('招聘信息汇总.xlsx')
    # symbol_cleaning = [" ", "\"", "“", "”", "*", "!", "！", "。", "."]

    # pd.set_option('display.max_rows', 2000)
    # pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.max_colwidth', 1000)

    file_data1 = file_data.groupby(['site', 'com_name']).agg({'com_name': ['count']})

    file_data2 = file_data.groupby(['site', 'type']).agg({'type': ['count']})

    file_data3 = file_data.groupby(['site', 'feature']).agg({'feature': ['count']})

    # #公司热度
    com_name_frequency = {}
    for city_name in city_list:
        com_name_frequency[city_name] = {}
        t_df = file_data1.loc[city_name, :]
        for t_name in t_df.index.values:
            t_name = t_name.strip()
            if com_name_frequency[city_name].get(t_name, -1) == -1:
                com_name_frequency[city_name][t_name] = int(t_df.loc[t_name, ('com_name', 'count')])
            else:
                com_name_frequency[city_name][t_name] += int(t_df.loc[t_name, ('com_name', 'count')])
    # print(com_name_frequency)
    # 去除值为1的键
    for city_name, content in copy.deepcopy(com_name_frequency).items():
        for com_name, count in content.items():
            if count == 1 and len(com_name) >= 8 or len(com_name) >= 14 and count == 2:
                com_name_frequency[city_name].pop(com_name)

    #
    # #公司类型热度
    com_type_frequency = {}

    for city_name in city_list:
        com_type_frequency[city_name] = {}
        t_df = file_data2.loc[city_name, :]

        for t_type in t_df.index.values:
            t_type_num = int(t_df.loc[t_type, ('type', 'count')])

            if isinstance(t_type, str):
                t_type_list = re.split('[ ,｜丨|，/、。;；!！+＋*~-]', t_type)
                t_type_list = list(set(t_type_list))

                for type_name in t_type_list:
                    if type_name != '':
                        if com_type_frequency[city_name].get(type_name, -1) == -1:
                            com_type_frequency[city_name][type_name] = t_type_num
                        else:
                            com_type_frequency[city_name][type_name] += t_type_num
    # print(com_type_frequency)

    # 去除值为1的键
    for city_name, content in copy.deepcopy(com_type_frequency).items():
        for type_name, count in content.items():
            if count == 1 or len(type_name) > 15:
                com_type_frequency[city_name].pop(type_name)

    #
    # #公司特点热度
    com_feature_frequency = {}

    for city_name in city_list:
        com_feature_frequency[city_name] = {}
        t_df = file_data3.loc[city_name, :]
        for t_feature in t_df.index.values:
            t_feature_num = int(t_df.loc[t_feature, ('feature', 'count')])
            if isinstance(t_feature, str):
                t_feature_list = re.split('[ ,｜丨|，/、。;；!！+＋*~-]', t_feature)
                t_feature_list = list(set(t_feature_list))
                for feature_name in t_feature_list:
                    if feature_name != '':
                        if com_feature_frequency[city_name].get(feature_name, -1) == -1:
                            com_feature_frequency[city_name][feature_name] = t_feature_num
                        else:
                            com_feature_frequency[city_name][feature_name] += t_feature_num
    # print(com_feature_frequency)

    # 去除值为1的键
    for city_name, content in copy.deepcopy(com_feature_frequency).items():
        for feature_name, count in content.items():
            if count == 1 or len(feature_name) > 15:
                com_feature_frequency[city_name].pop(feature_name)

    # #公司热度  总
    com_name_frequency_z = {}
    com_name = file_data.loc[:, 'com_name'].values
    for t_name in com_name:
        t_name = t_name.strip()
        if com_name_frequency_z.get(t_name, -1) == -1:
            com_name_frequency_z[t_name] = 1
        else:
            com_name_frequency_z[t_name] += 1

    # 公司类型热度 总
    com_type_frequency_z = {}
    com_type = file_data.loc[:, 'type'].values

    for t_type in com_type:
        if isinstance(t_type, str):
            t_type_list = re.split('[ ,｜丨|，/、。;；!！+＋~-]', t_type)
            t_type_list = list(set(t_type_list))
            for type_name in t_type_list:
                type_name = type_name.strip()
                if type_name != '':
                    if com_type_frequency_z.get(type_name, -1) == -1:
                        com_type_frequency_z[type_name] = 1
                    else:
                        com_type_frequency_z[type_name] += 1

    # 公司特点热度 总
    com_feature_frequency_z = {}
    com_feature = file_data.loc[:, 'feature'].values
    ii = 0
    for t_feature in com_feature:
        if isinstance(t_feature, str):
            t_feature_list = re.split('[ ,｜丨|，/、。;；!！+＋~-]', t_feature)
            t_feature_list = list(set(t_feature_list))
            for feature_name in t_feature_list:
                feature_name = feature_name.strip()
                if feature_name != '':
                    if com_feature_frequency_z.get(feature_name, -1) == -1:
                        com_feature_frequency_z[feature_name] = 1
                    else:
                        com_feature_frequency_z[feature_name] += 1

    print('*******', com_name_frequency)
    print('*******', com_type_frequency)
    print('*******', com_feature_frequency)
    return com_name_frequency, com_type_frequency, com_feature_frequency, com_name_frequency_z, com_type_frequency_z, com_feature_frequency_z


# 五
def feature_analyse5():
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 中文
    plt.rcParams['axes.unicode_minus'] = False  # 负号

    fj_file = open('../data/房价_total.json', 'r', encoding='utf-8')
    fj_sum = json.load(fj_file)  # 用.load
    fj_file.close()

    using_t_sort = [(i, j[-1][1]) for i, j in fj_sum.items()]
    using_t_sort = sorted(using_t_sort, key=lambda x: -x[1])
    # print(using_t_sort)
    city_names = [i[0] for i in using_t_sort]
    # print(city_names)

    total_pre_data = {}
    pre_len = 5  # 预测未来5个时间点的数据

    for city_name in city_names:

        fjbj_data = [i[1] for i in fj_sum[city_name]]
        fjbj_data = fjbj_data[-int(len(fjbj_data) * 4 / 5):]

        data = pd.DataFrame(columns=['house_price'])
        data.loc[:, 'house_price'] = np.array(fjbj_data)

        # print(data)
        # 原始序列
        # data.plot(color='g',style='-')
        # plt.xlabel('date')
        # plt.ylabel('house_price')
        # plt.show()

        # #一、平稳性检验
        #
        # 1、用自相关性图来检验
        # plot_acf(data,lags=50)
        # plt.show()
        # 自相关图既不是拖尾也不是截尾，从自相关图可以看出，自相关系数长期位于零轴一边，这是序列呈现单调趋势的明显特征
        # 故该序列是非平稳的
        #
        # #2、ADF检验
        t_res = adfuller(data)
        # print(t_res,type(t_res))
        # # 其返回值，第一个是adt检验的结果，即ADF Test result，简称为T值，表示t统计量。
        # # 第二个简称为p值，表示t统计量对应的概率值。
        # # 第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。
        #
        # # 判断是否平稳
        # # 第一点，1%、%5、%10不同程度拒绝原假设的统计值和前面所得t统计量的比较，若同时小于1%、5%、10%的统计值即说明非常好地拒绝该假设（表明为平稳数据）。
        # # 第二点，p值要求小于给定的显著水平，p值要小于0.05，等于0是最好的。
        #

        d = 0  # 差分数

        if t_res[1] >= 0.05:  # 若原始序列为非平稳序列，对于非平稳序列，通过差分运算使其转化为平稳序列
            # #二、差分运算
            #
            #
            # # 1、做一阶差分
            d1_data = data.diff(periods=1, axis=0)  # periods=1表示diff时隔一个数，axis=0按行，即上下
            # print(d1_data)
            d1_data = d1_data.dropna()  # 因为做差分所以第一个数为nan
            # print(d1_data)
            #
            # #原始序列一阶差分时序图
            # d1_data.plot()
            # plt.title('First-order difference')
            # plt.show()

            # # 其自相关图
            # plot_acf(d1_data,lags=50).show()
            # plt.show()
            #
            # #adf平稳性检验（主要）
            t_res = adfuller(d1_data)
            print('原始序列一阶差分的ADF检验结果为：', t_res)

            if t_res[1] >= 0.05:
                d2_data = d1_data.diff(periods=1, axis=0)
                d2_data = d2_data.dropna()
                # print(d2_data)

                # 原始序列二阶差分时序图
                # d2_data.plot()
                # plt.title('Second-order difference')
                # plt.show()
                # 其自相关图
                # plot_acf(d2_data, lags=50).show()
                # plt.show()

                # adf平稳性检验
                print('原始序列二阶差分的ADF检验结果为：', adfuller(d2_data))
                d = 2

            else:
                d2_data = d1_data
                d = 1
        else:
            d2_data = data

        # 三、白噪声检验，即纯随机性检验,当数据是纯随机数据时,再对数据进行分析就没有任何意义了
        # Ljung-Box检验(简称LB检验)其原假设和备择假设分别为HO:延迟期数小于或等于m期的序列之间相互独立(序列是白噪声); H1:延迟期数小于或等于m期的序列之间有相关性(序列不是白噪声)。
        # 如果p<0.05，拒绝原假设，说明原始序列存在相关性
        # 如果p>=0.05，接收原假设，说明原始序列独立，纯随机
        lags = [1, 2, 3]  # 延迟期数
        LB = acorr_ljungbox(d2_data, lags=lags, return_df=True)
        # print('\n'+str(d)+'阶差分序列的白噪声检验结果为：\n',LB)#返回统计量、P值
        # 可见所有p值都<0.05故说明二阶差分序列非白噪声

        pvalues = LB['lb_pvalue'].values.tolist()
        print(pvalues)
        p_judge = 1
        for i in pvalues:
            if i > 0.05:
                p_judge = 0

        if p_judge == 1:
            # 至此，获得平稳非白噪声序列。将该序列放入ARMA模型中获得预测模型。
            #
            # print(d)
            #
            # #四、模型定阶数
            # # #一般阶数不超过 len /10
            pmax = 5
            qmax = 5
            aic = []
            for p in range(pmax + 1):
                temp = []
                for q in range(qmax + 1):
                    try:
                        t_md = ARIMA(data, order=(p, d, q)).fit()
                        # ##残差检验
                        resid = t_md.resid
                        #
                        # LB检验
                        t_LB = acorr_ljungbox(resid, lags=1)
                        # print('残差序列的白噪声检验结果为：',t_LB )  # 返回统计量、P值
                        # #p>=0.05 ，故残差序列是白噪声
                        if t_LB['lb_pvalue'].values.tolist()[0] >= 0.05:
                            temp.append(t_md.aic)
                        else:
                            temp.append(None)
                    except:
                        temp.append(None)
                aic.append(temp)
                # print(temp)
            aic_matrix = pd.DataFrame(aic)  # 将其转换成Dataframe 数据结构
            # print(aic_matrix)
            p, q = aic_matrix.stack().idxmin()  # stack使列也变成行索引，aic_matrix.stack()的索引为复合索引，第一层索引为p，第二层为q
            # print(aic_matrix.stack())
            # print(u'AIC 最小的p值 和 q 值：%s,%s' %(p,q))

            # #五、模型评估、检验
            # # 建立模型后，需要对残差序列进行检验。若残差序列为白噪声序列，则说明时间序列中的有用信息已经被提取完毕，
            # # 剩下的全是随机扰动，是无法预测和使用的。
            # # 如果模型对原始序列解释性很好，那么这个新的序列与原始序列的差值（残差序列）应该是白噪声序列。
            #
            md = ARIMA(data, order=(p, d, q)).fit()
            #
            #
            # #模型预测
            forecastdata = md.forecast(pre_len)
            # print(type(forecastdata),'\n',forecastdata)

            data_pre = md.fittedvalues
            data_pre.loc[0] = float(data.loc[0])
            # print(data_pre) #已有样本数据的拟合数据

            ori_len = len(data_pre)
            for i in range(ori_len, ori_len + pre_len):
                data_pre.loc[i] = forecastdata.loc[i]
            # print(data_pre)

            # ax=plt.subplot(111)
            # plt.plot(data_pre,label='predict')
            # plt.plot(data,label='sample')
            # ax.set_xticks(ax.get_xticks()[::10]) #使x轴刻度显示得不密集
            #
            # plt.legend(loc='best')
            # plt.xlabel('date')
            # plt.ylabel('house_price')
            # plt.show()

            forecastdata = forecastdata.values.tolist()

            for i in range(len(forecastdata)):
                forecastdata[i] = round(float(forecastdata[i]), 4)

            total_pre_data[city_name] = forecastdata
        else:
            total_pre_data[city_name] = []

    print(total_pre_data)
    with open('房价_predict_total.json', 'w', encoding='utf-8') as f:
        json.dump(total_pre_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    feature_analyse3()

# from 数据特征工程1 import data_pretreatment,feature_analyse1
# Data_df0=pd.read_excel('招聘信息汇总.xlsx',sheet_name=0)
#
# Data_df1=data_pretreatment(Data_df0)
#
# city_name_list,mean_r_list,labels_explanations,labels_distribution=feature_analyse1(Data_df0,Data_df1)
#
# feature_analyse4(city_name_list)

# feature_analyse5()
