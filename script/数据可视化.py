# -*- coding:utf-8 -*-
# @File :数据可视化.py
# @Software: PyCharm
import json
import time
import random as rd

from pyecharts.commons.utils import JsCode

from 数据特征工程1 import data_pretreatment
from 数据特征工程1 import feature_analyse1
from 数据特征工程1 import feature_analyse2
from 数据特征工程2 import feature_analyse3
from 数据特征工程2 import feature_analyse4
from 数据特征工程2 import feature_analyse5
from pyecharts.globals import SymbolType

import pandas as pd
import numpy as np

from pyecharts import options as opts
from pyecharts.charts import Bar, Page, Scatter, WordCloud  # 导入柱状图
from pyecharts.charts import Line  # 折线
from pyecharts.charts import Pie  # 饼图
from pyecharts.charts import Map
from pyecharts.charts import Timeline
from pyecharts.charts import Boxplot
from pyecharts.charts import Geo
from pyecharts.charts import Liquid
from pyecharts.globals import ChartType


def figure1(city_name_list, mean_r_list, labels_explanations, labels_distribution):
    # print(labels_distribution['nop'])
    city_sample_count = np.array(labels_distribution['experience']).sum(axis=1)
    print(city_sample_count)
    max_s_count = city_sample_count.max()
    max_revenue = max(mean_r_list)

    timeline = Timeline()

    # colors=['skyblue','pink','red','yellow','blue','green','orange','purple']
    temp_title = {
        'experience': '城市招聘工作所需经验分布', 'attribute': '城市招聘工作性质分布',
        'nop': '城市招聘公司团队规模分布', 'qualification': '城市招聘工作所需学历分布'
    }
    temp_label = {
        'experience': '工作经验', 'attribute': '性质', 'nop': '规模', 'qualification': '学历'
    }

    for index, label in enumerate(['experience', 'attribute', 'nop', 'qualification']):
        t_label_data = np.array(labels_distribution[label])

        bar = Bar(init_opts=opts.InitOpts())  # 画布

        one_explanation = labels_explanations[label]
        for j, t_lb_exp in enumerate(one_explanation):
            bar.add_xaxis(city_name_list)  # 添加x轴的数据
            if t_lb_exp != '#':
                bar.add_yaxis(t_lb_exp, t_label_data[:, j].tolist(), stack=str(index), )  # 添加y轴的数据，y轴一定要有名字,柱状图1
                # print(j, t_label_data[:,j])
            else:
                bar.add_yaxis('#', t_label_data[:, j].tolist(), stack=str(index), is_selected=False,
                              label_opts=opts.LabelOpts(is_show=False, ), )
        # 全局配置
        bar.set_global_opts(title_opts=opts.TitleOpts(title=temp_title[label], pos_top='0', pos_left='35%', )  # 标题
                            , xaxis_opts=opts.AxisOpts(is_show=True)
                            , legend_opts=opts.LegendOpts(pos_top=29, border_color='transparent', item_gap=4)
                            , yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}"), name='人数',
                                                       max_=int(max_s_count * 1.15))
                            , datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=0, range_start=0, range_end=80, orient='vertical',
                                  pos_left='8px'),
                opts.DataZoomOpts(type_="slider", xaxis_index=0, is_show=True, orient='vertical', pos_left='8px'), ]
                            , toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                                , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                                   background_color='transparent')
                                                , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                                ))
                            )
        # 系列配置
        bar.set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="inside", )  # 显示在每个bar上的标签
                            , splitline_opts=opts.SplitLineOpts(is_show=True)
                            , itemstyle_opts=opts.ItemStyleOpts(opacity=0.7))
        bar.extend_axis(
            yaxis=opts.AxisOpts(
                interval=2000, min_=0, name='薪资', max_=int(max_revenue * 1.1)
            )
        )

        line = Line().add_xaxis(city_name_list).add_yaxis("平均薪资", mean_r_list, yaxis_index=1
                                                          , linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.9)
                                                          , label_opts=opts.LabelOpts(color='LightSeaGreen',
                                                                                      font_weight='bold')
                                                          , markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="max", name="max_mean_revenue"),
                      opts.MarkLineItem(type_="min", name="min_mean_revenue")],
                label_opts=opts.LabelOpts(color='rgba(32, 178, 170,0.85)'))
                                                          )
        bar.overlap(line)

        timeline.add(bar, time_point=temp_label[label])

    timeline.add_schema(is_auto_play=True, play_interval=6800
                        , label_opts=opts.LabelOpts(font_weight='bold'))

    timeline.render('.\\feature_t1.html')
    return timeline


def figure2(difficulty_df0):
    city_name_list = np.unique(difficulty_df0.loc[:, 'site'].values).tolist()
    city_difficulty_dict = {}
    for i in range(len(difficulty_df0)):
        city_name = difficulty_df0.loc[i, 'site']
        if city_difficulty_dict.get(city_name, -1) == -1:
            city_difficulty_dict[city_name] = []
            city_difficulty_dict[city_name].append(np.round(difficulty_df0.loc[i, 'degree of difficulty'], 4))
        else:
            city_difficulty_dict[city_name].append(np.round(difficulty_df0.loc[i, 'degree of difficulty'], 4))

    c = Boxplot()
    c.add_xaxis(city_name_list)
    ydata = []
    mean_difficulty = []

    for k in city_name_list:
        ydata.append(city_difficulty_dict[k])
        mean_difficulty.append(np.array(city_difficulty_dict[k]).mean().round(2))

    c.add_yaxis("难度系数分布(箱线图)", c.prepare_data(ydata))
    c.set_global_opts(title_opts=opts.TitleOpts(title="城市招聘工作就职难度系数分布", pos_top='0', pos_left='35%')
                      , legend_opts=opts.LegendOpts(pos_top=29, border_color='transparent', item_gap=5)
                      , datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=20, range_end=80),
            opts.DataZoomOpts(type_="slider", xaxis_index=0, is_show=True), ]
                      , yaxis_opts=opts.AxisOpts(name='难度系数')
                      , toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                               background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                            ))
                      )

    scatter = Scatter()
    scatter.add_xaxis(xaxis_data=city_name_list)
    scatter.add_yaxis('难度系数分布(平均值)', mean_difficulty
                      , markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(type_="max", name="max_mean_difficulty"),
                  opts.MarkLineItem(type_="min", name="min_mean_difficulty")]
            , label_opts=opts.LabelOpts(color='rgba(5, 249, 117,0.9)', formatter='{c}'),
            linestyle_opts=opts.LineStyleOpts(color="rgba(5, 249, 117,0.9)", type_='dashed'))
                      )

    c.overlap(scatter)

    c.render('.\\feature_t2.html')
    return c


def figure3(class_res_df, class_explanation):
    target_df0 = class_res_df.loc[:, ['site', 'revenue_class']]
    target_df1 = target_df0.groupby(['revenue_class', 'site']).agg({'site': ['count']})
    # print(target_df1)
    # print(target_df1.index.values)

    visual_data = {}

    for tuple_t in target_df1.index.values:  # tuple的第一个值为类别，第二个值为城市
        class_ = str(tuple_t[0])
        city = tuple_t[1]
        if visual_data.get(class_, -1) == -1:
            visual_data[class_] = {}
        visual_data[class_][city] = target_df1.loc[tuple_t, ('site', 'count')]
    print(visual_data)

    revenue_meaning = ['低薪资', '中低薪资', '中薪资', '中高薪资', '高薪资', '超高薪资']
    pie_data_sum = {}
    for i_t in range(len(class_explanation)):
        i_t_s = str(i_t)
        pie_data_sum[i_t_s] = [[i, int(j)] for i, j in visual_data[i_t_s].items()]

    subtt = '由里到外薪资等级依次升高\n' + '其中平均薪资(/元)为：\n\n'
    for i_t in range(len(class_explanation)):
        tuple_t = class_explanation[i_t]
        subtt += revenue_meaning[i_t] + ':' + str(tuple_t[1]) + '\n'

    pie = Pie()

    s0 = class_explanation[0][0]
    pie.add(revenue_meaning[0], pie_data_sum[s0], radius=['3%', '19%'], center=['50%', '50%'], rosetype='area',
            label_opts=opts.LabelOpts(is_show=False))
    s1 = class_explanation[1][0]
    pie.add(revenue_meaning[1], pie_data_sum[s1], radius=['21%', '29%'], center=['50%', '50%'],
            label_opts=opts.LabelOpts(is_show=False, position='inside', formatter="{d}%", font_size=8,
                                      font_weight='lighter'))
    s2 = class_explanation[2][0]
    pie.add(revenue_meaning[2], pie_data_sum[s2], radius=['35%', '42%'], center=['50%', '50%'],
            label_opts=opts.LabelOpts(is_show=True, position='inside', formatter="{d}%", font_size=8,
                                      font_weight='lighter'))
    s3 = class_explanation[3][0]
    pie.add(revenue_meaning[3], pie_data_sum[s3], radius=['47%', '54%'], center=['50%', '50%'],
            label_opts=opts.LabelOpts(is_show=True, position='inside', formatter="{d}%", font_size=10,
                                      font_weight='lighter'))
    s4 = class_explanation[4][0]
    pie.add(revenue_meaning[4], pie_data_sum[s4], radius=['59%', '66%'], center=['50%', '50%'],
            label_opts=opts.LabelOpts(is_show=True, position='inside', formatter="{d}%", font_size=10))
    s5 = class_explanation[5][0]
    pie.add(revenue_meaning[5], pie_data_sum[s5], radius=['75%', '95%'], center=['50%', '50%'], rosetype='area',
            label_opts=opts.LabelOpts(is_show=True, formatter="{d}%", font_size=12, font_weight='bold'))

    pie.set_global_opts(title_opts=opts.TitleOpts(title="各城市各类薪资水平的工作数占比"
                                                  )
                        , legend_opts=opts.LegendOpts(is_show=True, textstyle_opts=opts.TextStyleOpts(font_size=10, ),
                                                      pos_right='1%', orient='vertical'
                                                      , type_='scroll', background_color='transparent',
                                                      border_color='black', inactive_color='rgb(187, 187, 162)')
                        , tooltip_opts=opts.TooltipOpts(formatter="{a}_{b}:{c}份_{d}%")
                        , toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                               background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                            ))
                        )

    pie.render('.\\feature_t3.html')
    return pie


def figure4(com_name_frequency, com_type_frequency, com_feature_frequency, com_name_frequency_z, com_type_frequency_z,
            com_feature_frequency_z, city_list):
    # 词云图
    timeline = Timeline(init_opts=opts.InitOpts(width='1150px', height='555px'))

    for city_name in city_list:  # 各城市
        subt = ' ' * 21 + city_name
        if city_name == '乌鲁木齐' or city_name == '呼和浩特':
            subt = ' ' * 16 + city_name
        if city_name == '哈尔滨' or city_name == '石家庄':
            subt = ' ' * 19 + city_name

        data0 = [(i, j) for i, j in com_name_frequency[city_name].items()]
        data1 = [(i, j) for i, j in com_type_frequency[city_name].items()]
        data2 = [(i, j) for i, j in com_feature_frequency[city_name].items()]

        cloud = WordCloud()

        cloud.add(series_name=city_name + ":公司热度", data_pair=data0, word_size_range=[11.5, 21]
                  , textstyle_opts=opts.TextStyleOpts(), width='350px', height='415px', pos_left='1px', pos_top='10%', )
        cloud.add(series_name=city_name + ":公司类型热度", data_pair=data1, word_size_range=[12, 23]
                  , textstyle_opts=opts.TextStyleOpts(), width='350px', height='415px', pos_left='385px',
                  pos_top='10%', )

        cloud.add(series_name=city_name + ":公司特点热度", data_pair=data2, word_size_range=[12, 22]
                  , textstyle_opts=opts.TextStyleOpts(), width='350px', height='415px', pos_left='755px',
                  pos_top='10%', )

        cloud.set_global_opts(
            title_opts=opts.TitleOpts(
                title="公司、类型、特点词云图", title_textstyle_opts=opts.TextStyleOpts(font_size=19), pos_top='0%',
                pos_left='38.5%', subtitle=subt, subtitle_textstyle_opts=opts.TextStyleOpts(font_size=14)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
            legend_opts=opts.LegendOpts(border_color='transparent'),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                                , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                                   background_color='transparent')
                                                , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                                ))
        )

        timeline.add(cloud, time_point=city_name)

    # #汇总
    # data0 = [ (i,j) for i,j in com_name_frequency_z.items() ]
    # data1 = [(i, j) for i, j in com_type_frequency_z.items()]
    # data2 = [(i, j) for i, j in com_feature_frequency_z.items()]
    #
    # cloud = WordCloud( )
    # cloud.add(series_name= "公司热度_汇总", data_pair=data0, word_size_range=[5, 16.5]
    #           , textstyle_opts=opts.TextStyleOpts(), width='350px', height='350px', pos_left='1px', pos_top='12%', )
    # cloud.add(series_name=  "公司类型热度_汇总", data_pair=data1, word_size_range=[5, 20]
    #           , textstyle_opts=opts.TextStyleOpts(), width='350px', height='350px', pos_left='360px', pos_top='12%', )
    #
    # cloud.add(series_name= "公司特点热度_汇总", data_pair=data2, word_size_range=[5, 20]
    #           , textstyle_opts=opts.TextStyleOpts(), width='350px', height='350px', pos_left='710px', pos_top='12%', )
    # timeline.add(cloud, time_point='汇总')
    #
    # #汇总(无字节)
    # com_name_frequency_z['字节跳动']=0
    # data0 = [ (i,j) for i,j in com_name_frequency_z.items() ]
    # data1 = [(i, j) for i, j in com_type_frequency_z.items()]
    # data2 = [(i, j) for i, j in com_feature_frequency_z.items()]
    #
    # cloud = WordCloud( )
    # cloud.add(series_name= "公司热度_汇总(无字节跳动)", data_pair=data0, word_size_range=[5, 20]
    #           , textstyle_opts=opts.TextStyleOpts(), width='350px', height='350px', pos_left='1px', pos_top='12%', )
    # cloud.add(series_name=  "公司类型热度_汇总", data_pair=data1, word_size_range=[5, 20]
    #           , textstyle_opts=opts.TextStyleOpts(), width='350px', height='350px', pos_left='360px', pos_top='12%', )
    #
    # cloud.add(series_name= "公司特点热度_汇总", data_pair=data2, word_size_range=[5, 20]
    #           , textstyle_opts=opts.TextStyleOpts(), width='350px', height='350px', pos_left='710px', pos_top='12%', )
    # timeline.add(cloud, time_point='汇总(无字节跳动)')

    timeline.add_schema(is_auto_play=False, play_interval=37000, orient='horizontal', width='700px', pos_left='15%',
                        pos_bottom='0%')
    timeline.render('.\\feature_t4.html')

    return timeline


def figure5(city_list):  # map热力图
    jy_df0 = pd.read_excel('各城市就业与失业人数.xlsx', sheet_name=0)

    # 城镇单位从业人数
    jy_num_com = jy_df0.loc[:, ['城市', '城镇单位从业人数（万人）']].values
    # print(jy_num_com)
    jy_num_com_values = []

    for i, j in jy_num_com:
        if str(j) == '无':
            continue
        else:
            jy_num_com_values.append((i, float(j)))
    jy_num_t = [i[1] for i in jy_num_com_values]
    jy_num_min1 = np.array(jy_num_t).min()
    jy_num_max1 = np.array(jy_num_t).max()

    # 城镇私营和个体从业人员
    jy_num_per = jy_df0.loc[:, ['城市', '城镇私营和个体从业人员（万人）']].values
    # print(jy_num_per)
    jy_num_per_values = []

    for i, j in jy_num_per:
        if str(j) == '无':
            continue
        else:
            jy_num_per_values.append((i, float(j)))
    jy_num_t = [i[1] for i in jy_num_per_values]
    jy_num_min2 = np.array(jy_num_t).min()
    jy_num_max2 = np.array(jy_num_t).max()

    timeline = Timeline()

    c1 = Geo()
    c1.add_schema(maptype="china", is_roam=False)
    c1.add("城镇单位从业人数", jy_num_com_values, type_=ChartType.EFFECT_SCATTER, label_opts=opts.LabelOpts(is_show=False)
           , symbol_size=13, effect_opts=opts.EffectOpts(trail_length=0.5, scale=3))
    c1.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=jy_num_max1, min_=jy_num_min1, pos_left='15.2%', pos_bottom='28%'
                                          , range_color=["aqua", 'rgb(99, 255, 156)', 'yellow', 'orange', "red"],
                                          range_text=['城镇单位从业人数(万人)', '']),
        title_opts=opts.TitleOpts(title="城镇单位从业人数", pos_left='15%', pos_top='11%'),
        legend_opts=opts.LegendOpts(pos_top='10%'),

        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_='png', background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)
                                            , data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False)
                                            ), pos_top='10%')
        )

    c2 = Geo()
    c2.add_schema(maptype="china", is_roam=False)
    c2.add("城镇私营和个体从业人数", jy_num_per_values, type_=ChartType.EFFECT_SCATTER, label_opts=opts.LabelOpts(is_show=False)
           , symbol_size=13)
    c2.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=jy_num_max2, min_=jy_num_min2, pos_left='13.8%', pos_bottom='28%'
                                          , range_color=["aqua", 'rgb(99, 255, 156)', 'yellow', 'orange', "red"],
                                          range_text=['城镇私营和个体从业人数(万人)', '']),
        title_opts=opts.TitleOpts(title="城镇私营和个体从业人数", pos_left='15%', pos_top='11%'),
        legend_opts=opts.LegendOpts(pos_top='10%'),

        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_='png', background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)
                                            , data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False)
                                            ), pos_top='10%')
        )

    timeline.add(c1, time_point='城镇单位从业人数')
    timeline.add(c2, time_point='城镇私营和个体从业人数')

    timeline.add_schema(is_auto_play=True, play_interval=23000
                        , pos_bottom='27%', width='350px', pos_left='35%')
    timeline.render("geo_heatmap.html")

    return timeline


def figure6(city_list):  # 水球图
    hp_df0 = pd.read_excel('各城市房价收入比.xlsx', sheet_name=0)

    income_radio = hp_df0.loc[:, ['城市', '收入比']].values
    income_radio_values = {}

    for i, j in income_radio:
        if isinstance(j, str):
            income_radio_values[i] = 0
        else:
            income_radio_values[i] = float(j)
    # print(income_radio_values)
    liquid_data = []
    for city_name in city_list:
        liquid_data.append((city_name, income_radio_values[city_name]))
    liquid_data = sorted(liquid_data, key=lambda x: -x[1])

    timeline = Timeline()
    index = 0
    for l_data_t in liquid_data:
        index += 1
        c = Liquid()
        city_name = l_data_t[0]
        liquid_t_v = l_data_t[1] / 100
        bl_n = 0
        if index <= 3:
            bl_n = 6
        elif index <= 10:
            bl_n = 5
        elif index <= 15:
            bl_n = 4
        else:
            bl_n = 3

        c.add("平均收入比", [liquid_t_v for i in range(0, bl_n)])
        ftw = 'bold'
        if index <= 3:
            ftw = 'bolder'

        psl = '41.5%'
        subt = '(每户平均住房总价/每户平均年总收入)/100'
        if city_name == '乌鲁木齐' or city_name == '呼和浩特':
            psl = '38%'
            subt = '   (每户平均住房总价/每户平均年总收入)/100'
        if city_name == '哈尔滨' or city_name == '石家庄':
            psl = '40%'

        c.set_global_opts(title_opts=opts.TitleOpts(title=city_name + "平均收入比", pos_left=psl, pos_top='14%'
                                                    , title_textstyle_opts=opts.TextStyleOpts(font_size=23,
                                                                                              font_weight=ftw)

                                                    , subtitle_textstyle_opts=opts.TextStyleOpts(font_size=10))
                          , legend_opts=opts.LegendOpts(is_show=False)
                          , toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                                , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                                   background_color='transparent')
                                                , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                                ), pos_top='13%', pos_right='13%')
                          )

        timeline.add(c, time_point=city_name)

    timeline.add_schema(is_auto_play=True, play_interval=8500
                        , label_opts=opts.LabelOpts(font_weight='bold'), pos_bottom='70px')
    timeline.render("liquid_base.html")

    return timeline


def figure7():  # 折线/面积图
    fj_file = open('../data/房价_total.json', 'r', encoding='utf-8')
    fj_sum = json.load(fj_file)  # 用.load
    fj_file.close()

    using_t_sort = [(i, j[-1][1]) for i, j in fj_sum.items()]
    using_t_sort = sorted(using_t_sort, key=lambda x: -x[1])
    city_names = [i[0] for i in using_t_sort]
    # print(city_names)

    time_x = [i[0] for i in fj_sum['北京']]
    time_gap = time_x[-1] - time_x[-2]

    time_st = len(time_x) - 1
    for i in range(5):
        time_x.append(time_x[time_st] + time_gap)
        time_st += 1

    for i in range(len(time_x)):
        t_t = time_x[i]
        t_t = time.localtime(t_t / 1000)
        time_x[i] = time.strftime("%Y-%m-%d", t_t)
    print(time_x)

    fj_pre_file = open('房价_predict_total.json', 'r', encoding='utf-8')
    fj_pre_sum = json.load(fj_pre_file)  # 用.load
    fj_pre_file.close()
    pre_num = len(fj_pre_sum['北京'])

    timeline = Timeline()
    c_top = Line()
    c_top.add_xaxis(time_x)
    c_ot = Line()
    c_ot.add_xaxis(time_x)

    ot_ct = 0
    ot_index = 0
    judge = True
    for index, city_name in enumerate(city_names):
        fj_data_t = fj_sum[city_name]
        fj_data_t = [i[1] for i in fj_data_t]
        fj_data_t.extend(fj_pre_sum[city_name])

        # print(city_name,fj_data_t)
        rd.seed(index)
        rdcolor = "rgb(" + str(rd.randint(0, 125) * 2) + "," + str(rd.randint(0, 125) * 2) + "," + str(
            rd.randint(0, 100) * 2) + ")"

        if index < 4:

            c_top.add_yaxis(series_name=city_name, y_axis=fj_data_t, color=rdcolor
                            , label_opts=opts.LabelOpts(is_show=False)
                            , linestyle_opts=opts.LineStyleOpts(width=5),
                            areastyle_opts=opts.AreaStyleOpts(opacity=0.7), )
        else:
            ot_index += 1
            if ot_ct >= 13 or ot_index % 2 == 0 and ot_ct >= 6:
                judge = False
            else:
                ot_ct += 1
                judge = True
            c_ot.add_yaxis(series_name=city_name, y_axis=fj_data_t, color=rdcolor
                           , label_opts=opts.LabelOpts(is_show=False), symbol_size=2, is_selected=judge)
    #

    c_top.set_global_opts(title_opts=opts.TitleOpts(title="城市近两年每周平均房价", pos_top='2px', pos_left='37%',
                                                    subtitle='          更新时间：' + time_x[-(pre_num + 1)])
                          , yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}"), name='元/平方米',
                                                     min_=41000, )

                          , tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type='cross')
                          , datazoom_opts=[
            opts.DataZoomOpts(type_="inside", xaxis_index=0, range_start=50, range_end=100, orient='vertical',
                              pos_left='15px'),
            opts.DataZoomOpts(type_="slider", xaxis_index=0, is_show=True, orient='vertical', pos_left='2px'), ]
                          , legend_opts=opts.LegendOpts(is_show=True, textstyle_opts=opts.TextStyleOpts(font_size=10,
                                                                                                        font_weight='bold')
                                                        , pos_right='1px', orient='vertical', type_='scroll',
                                                        border_color='black', inactive_color='rgb(187,187,162)')
                          , toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                               background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                            ))

                          )
    c_top.set_series_opts(
        markarea_opts=opts.MarkAreaOpts(
            is_silent=True,
            data=[
                opts.MarkAreaItem(
                    name="预测区间", x=(time_x[-pre_num], time_x[-1]),
                    label_opts=opts.LabelOpts(font_family='KaiTi', font_size=14, font_weight='bold'),
                    itemstyle_opts=opts.ItemStyleOpts(color="rgba(187, 198, 232,0.45)", opacity=0.4),
                ),

            ],
        ),
    )

    c_ot.set_global_opts(title_opts=opts.TitleOpts(title="城市近两年每周平均房价", pos_top='2px', pos_left='37%',
                                                   subtitle='          更新时间：' + time_x[-(pre_num + 1)])
                         , yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}"), name='元/平方米',
                                                    min_=5200, interval=2500)

                         , tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type='cross')
                         , datazoom_opts=[
            opts.DataZoomOpts(type_="inside", xaxis_index=0, range_start=50, range_end=100, orient='vertical',
                              pos_left='15px'),
            opts.DataZoomOpts(type_="slider", xaxis_index=0, is_show=True, orient='vertical', pos_left='2px'), ]
                         , legend_opts=opts.LegendOpts(is_show=True, textstyle_opts=opts.TextStyleOpts(font_size=10,
                                                                                                       font_weight='bold')
                                                       , pos_right='1px', orient='vertical', type_='scroll',
                                                       border_color='black', inactive_color='rgb(187,187,162)')
                         , toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                               background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                            ))

                         )

    timeline.add(c_top, time_point='排名前四城市')
    timeline.add(c_ot, time_point='其他城市(点击图例按钮可进行选择)')

    timeline.add_schema(is_auto_play=True, play_interval=7500
                        , label_opts=opts.LabelOpts(font_weight='bold'))

    timeline.render('.\\feature_t7.html')

    return timeline


def figure8():
    lose_job_df0 = pd.read_excel('各城市就业与失业人数.xlsx', sheet_name=0)

    lj_data = lose_job_df0.loc[:, ['城市', '城镇登记失业人员数（万人）']].values

    lj_data = sorted(lj_data, key=lambda x: x[1])

    lj_city = [i for i, j in lj_data]
    lj_num = [float(j) for i, j in lj_data]

    c = Bar()

    c.add_xaxis(lj_city)
    c.add_yaxis('城市登记失业人员数', lj_num)

    c.reversal_axis()
    c.set_series_opts(label_opts=opts.LabelOpts(position="right", font_size=12, font_style='oblique', formatter='{c}万人')
                      , itemstyle_opts={
            "normal": {
                "color": JsCode(
                    """
                    function (params) {
                        if (params.value >= 20) {
                            return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{
                                offset: 0,
                                color: 'rgba(255, 59, 59, 1)' 
                            }, {
                                offset: 1,
                                color: 'rgba(255,146, 59, 1)'
                            }], false);
                        } else if (params.value >= 15 && params.value < 20) {
                            return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{
                                offset: 0,
                                color: 'rgba(255,240, 59, 1)'
                            }, {
                                offset: 1,
                                color: 'rgba(255,146, 59, 1)'
                            }], false);
                        }else if (params.value >= 10 && params.value < 15) {
                            return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{
                                offset: 0,
                                color: 'rgba(255,240, 59, 1)'
                            }, {
                                offset: 1,
                                color: 'rgba(137, 255, 59, 1)'
                            }], false);
                        }else if (params.value >= 5 && params.value <10) {
                            return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{
                                offset: 0,
                                color: 'rgba(59, 255, 233, 1)'
                            }, {
                                offset: 1,
                                color: 'rgba(137, 255, 59, 1)'
                            }], false);
                        }
                        return new echarts.graphic.LinearGradient(1, 0, 0, 0, [{
                                offset: 0,
                                color: 'rgba(59, 255, 233, 1)'
                            }, {
                                offset: 1,
                                color: 'rgba(59, 205, 255, 1)'
                            }], false);
                    }
                    """
                ),
                "barBorderRadius": [10, 60, 60, 10],
                "shadowColor": "rgb(0, 160, 221)",
            }
        }
                      )
    c.set_global_opts(title_opts=opts.TitleOpts(title="城市登记失业人员数", pos_top='2px', pos_left='41.5%')
                      , xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}"), name='人数(万人)')
                      ,
                      yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_weight='bold', margin=25), name='城市')
                      , legend_opts=opts.LegendOpts(is_show=True,
                                                    textstyle_opts=opts.TextStyleOpts(font_size=10, font_weight='bold')
                                                    , pos_right='1px', orient='vertical', type_='scroll',
                                                    border_color='tansparent')

                      , datazoom_opts=[
            opts.DataZoomOpts(type_="inside", yaxis_index=0, range_start=40, range_end=100, orient='vertical',
                              pos_left='15px'),
            opts.DataZoomOpts(type_="slider", yaxis_index=0, is_show=True, orient='vertical', pos_right='45px',
                              pos_top='45px'), ]
                      , toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                                            , save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
                                                                                               background_color='transparent')
                                            , data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False)

                                            ), pos_left='5%')
                      )
    c.render("bar_reversal_axis.html")

    return c


#
# #
# Data_df0=pd.read_excel('招聘信息汇总.xlsx',sheet_name=0)
#
# Data_df1=data_pretreatment(Data_df0)
#
# city_name_list,mean_r_list,labels_explanations,labels_distribution=feature_analyse1(Data_df0,Data_df1)
# figure1(city_name_list,mean_r_list,labels_explanations,labels_distribution)
#
# difficulty_df=feature_analyse2(Data_df1)
# # figure2(difficulty_df)
#
# class_res_df,class_explanation=feature_analyse3()
# # figure3(class_res_df,class_explanation)
#
# com_name_frequency,com_type_frequency,com_feature_frequency  ,com_name_frequency_z,com_type_frequency_z,com_feature_frequency_z=feature_analyse4(city_name_list)
# # figure4(com_name_frequency,com_type_frequency,com_feature_frequency ,com_name_frequency_z,com_type_frequency_z,com_feature_frequency_z,city_name_list)
#
# # figure5(city_name_list)
#
# # figure6(city_name_list)
#
# # feature_analyse5()
# # figure7()
#
# # figure8( )

def big_title():
    line = (
        Line()
            .add_xaxis([None])
            .add_yaxis("", [None])
            .set_global_opts(
            title_opts=opts.TitleOpts(title="择      业      通      航",
                                      pos_left='center',
                                      title_textstyle_opts=opts.TextStyleOpts(font_size=52, font_weight='bolder',
                                                                              font_family='KaiTi'),
                                      pos_top='1%'),
            yaxis_opts=opts.AxisOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(is_show=False),
            legend_opts=opts.LegendOpts(is_show=False))
    )

    return line


page = Page(layout=Page.DraggablePageLayout, page_title="择业通航")

# # #在页面中添加图表
# page.add(
#     # big_title(),
#     figure5(city_name_list),
#     figure6(city_name_list),
#
#     figure1(city_name_list,mean_r_list,labels_explanations,labels_distribution),
#     figure2(difficulty_df),
#     figure8(),
#     figure3(class_res_df,class_explanation),
#     figure4(com_name_frequency,com_type_frequency,com_feature_frequency ,com_name_frequency_z,com_type_frequency_z,com_feature_frequency_z,city_name_list),
#
#     figure7(),
#
# )
#
# page.render('test.html')


page.save_resize_html('test.html', cfg_file='chart_config.json', dest='择业通航.html')
