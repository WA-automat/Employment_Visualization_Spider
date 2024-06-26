import json
import pandas as pd
from pmdarima.arima import auto_arima

if __name__ == '__main__':
    predict_num = 4
    new_data = {}
    df = pd.read_excel('../data/保险支出.xlsx', sheet_name='Sheet2', index_col=0, header=0)
    name_list = df.columns
    for index in name_list.to_list():
        insurance_data = df[index].values.tolist()
        model = auto_arima(insurance_data)
        forcast = model.predict(n_periods=predict_num)
        insurance_data += forcast.tolist()
        new_items = [list(item) for item in zip(list(range(2006, 2026)), insurance_data)]
        new_data[index] = new_items

    with open("../data/insurance_2.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
