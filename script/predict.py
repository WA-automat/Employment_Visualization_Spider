import json
from pmdarima.arima import auto_arima

with open("../data/房价_total.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

if __name__ == '__main__':
    new_data = {}
    predict_num = 3
    cities = data.keys()
    for city in cities:
        items = data[city]
        time_series = [item[0] for item in items]
        house_price = [item[1] for item in items]
        model = auto_arima(house_price)
        forcast = model.predict(n_periods=predict_num)
        house_price += forcast.tolist()
        for i in range(predict_num):
            time_series.append(time_series[len(time_series) - 1] + time_series[1] - time_series[0])
        new_items = [list(item) for item in zip(time_series, house_price)]
        new_data[city] = new_items

    with open("../data/房价_total_new.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
