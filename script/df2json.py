import pandas as pd

if __name__ == '__main__':

    df = pd.read_excel("../data/招聘信息汇总.xlsx", sheet_name='sheet_1')
    df_afterProcessing = pd.read_excel("../data/招聘信息_afterProcessing.xlsx", sheet_name='sheet_1')

    df.to_json("../data/招聘信息汇总.json", orient='records', force_ascii=False)
    df_afterProcessing.to_json("../data/招聘信息_afterProcessing.json", orient='records', force_ascii=False)
