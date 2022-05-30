import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
from keras.models import load_model
import plotly.offline as py
import pandas_datareader as web
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = load_model('mmy_best_model.hdf5')
model.make_predict_function() 
predicted_btc_price_test_data=np.array([])

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    start = dt.datetime.now()- dt.timedelta(days=60)
    end = dt.datetime.now()

    df = web.DataReader('BTC-USD', 'yahoo', start, end)
    df = df.rename_axis('Time').reset_index()
    df_datetype =df.astype({'Volume': 'float64'})
    del df_datetype["Adj Close"]


    df_datetype['date'] = pd.to_datetime(df_datetype['Time'],unit='s').dt.date



    group = df_datetype.groupby('date')

    btc_closing_price_groupby_date = group['Close'].mean()

    prediction_days = 60

    df_test= btc_closing_price_groupby_date[len(btc_closing_price_groupby_date)-prediction_days:].values.reshape(-1,1)

    scaler_test = MinMaxScaler(feature_range=(0, 1))
    scaled_test = scaler_test.fit_transform(df_test)

    def dataset_generator_lstm(dataset, look_back=5):
 
        dataX, dataY = [], []
        
        for i in range(len(dataset) - look_back):
            window_size_x = dataset[i:(i + look_back), 0]
            dataX.append(window_size_x)
            dataY.append(dataset[i + look_back, 0]) 
        return np.array(dataX), np.array(dataY)


    testX, testY = dataset_generator_lstm(scaled_test)

    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1 ))


    valluu=int(next(request.form.values()))
    lookback_period =  valluu#burdaki 10 u değiştir


    testX_last_5_days = testX[testX.shape[0] - lookback_period :  ]

    predicted_5_days_forecast_price_test_x = []

    for i in range(valluu):  #burdaki 10 u değiştir
      predicted_forecast_price_test_x = model.predict(testX_last_5_days[i:i+1])#chechkpoint önümüzceki gün için tahmini yapıyor
      
      predicted_forecast_price_test_x = scaler_test.inverse_transform(predicted_forecast_price_test_x.reshape(-1, 1))#scale edilmiş veri olduğu için yeniden eski haline getiriyoruz

      predicted_5_days_forecast_price_test_x.append(predicted_forecast_price_test_x)#yeni veriyi eski verilerin yanına ekliyoruz
  
#plot işlemi kolay olsun diye aşşağıdaki işlemleri yapıyorum
    predicted_5_days_forecast_price_test_x = np.array(predicted_5_days_forecast_price_test_x)

    predicted_5_days_forecast_price_test_x = predicted_5_days_forecast_price_test_x.flatten()
    global predicted_btc_price_test_data
    predicted_btc_price_test_data = predicted_btc_price_test_data.flatten()

    predicted_btc_test_concatenated = np.concatenate((predicted_btc_price_test_data, predicted_5_days_forecast_price_test_x))


    plt.figure(figsize=(16,7))

    plt.plot(predicted_btc_test_concatenated, 'r', marker='.', label='Predicted Test')

    #plt.plot(test_actual, marker='.', label='Actual Test')

    plt.legend()

    plt.show()

    return render_template('index.html', prediction_text=plt.show())

    #return render_template('index.html', prediction_text='Salary is {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
