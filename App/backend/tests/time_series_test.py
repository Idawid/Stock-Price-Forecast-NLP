# Test functions. Remove in subsequent commit !!!!!!
import json
import yfinance as yf

from backend.time_series.time_series_model import TimeSeriesModel

N_EPOCHS = 1000
N_NEURONS = 512
N_LAYERS = 4
N_STACKS = 32
HORIZON = 1
WINDOW_SIZE = 7  # 8 with sentiment 7 without

INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON

test_ticker = "AAPL"
test_from = "2023-11-01"
test_to = "2023-12-04"

timeSeries = TimeSeriesModel()
df_stock = yf.download(test_ticker, start=test_from, end=test_to)
df_prep = timeSeries.make_windowed_data_without_sentiment(df_stock, 7, 1)  ##
X_train, y_train, X_test, y_test = timeSeries.test_training_split(df_prep)
train_dataset, test_dataset = timeSeries.prepare_data_for_training(X_train, y_train, X_test, y_test, 1024)
dataset_all, X_all, y_all = timeSeries.prepare_data_for_prediction(df_prep, 1024, 7)
model = timeSeries.nbeatsModel(INPUT_SIZE, THETA_SIZE, HORIZON, N_NEURONS, N_LAYERS, N_STACKS, train_dataset, N_EPOCHS,
                               test_dataset)
future_forecast = timeSeries.make_future_forecast_without_sentiment(
    values=y_all,
    model=model,
    into_future=7,
    window_size=WINDOW_SIZE)
to_send = [str(x) for x in future_forecast]

json_data = json.dumps(to_send)
model.save('saved_model/my_model')
print(json_data)
