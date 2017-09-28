from bottle import route,get, run, template, request, post,response, HTTPResponse
from bottle import TEMPLATE_PATH, static_file
from sklearn.externals import joblib
import subprocess,os
import numpy as np
import keras
import json
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Activation, Dropout,LSTM
from keras import regularizers
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

@route('/css/<filename>')
def css_dir(filename):
    print(filename)
    print(BASE_DIR + '\\views\static\css\\')
    print(static_file(filename, root=BASE_DIR + '\\views\static\css\\'))
    return static_file(filename, root=BASE_DIR + "\\views\static\css\\")

@get("/")
def hello():
    return template("free")  #views/index.htmlを返す

@get("/train")
def training():
    d_ = "./data"
    m_ = "./model"
    command = "python data_get.py"
    subprocess.call(command)
    return template("menu")

@get("/menu")
def menu():
    return template("menu")

@get("/ridge")
def predict():
    predict_list = []
    selection = request.query.getall("model")
    
    d_ = ".\data"
    m_ = ".\model"

    d = date.today()
    str_d = str(d)
    
    y_test_data_file_name = str_d + "_y_test_data"
    X_test_data_file_name = str_d + "_X_test_data"

    x_test_data = np.load(os.path.join(d_, X_test_data_file_name+".npy"))
    y_test_data = np.load(os.path.join(d_, y_test_data_file_name+".npy"))
    x = np.arange(0, y_test_data.shape[0]-1)
    x = x.tolist()
    y_test_data = y_test_data.tolist()
    model = joblib.load(os.path.join(m_,"ridge.pkl"))
    
    predicted_ridge = model.predict(x_test_data)
    predicted_ridge = predicted_ridge.tolist()
    
    answer = division(y_test_data[-2], predicted_ridge[-1])
    
    predicted_last = predicted_ridge[-1]
    y_last = y_test_data[-1]
    
    predicted_ridge = predicted_ridge[:-1]
    y_test_data = y_test_data[:-1]
    
    return template("plots", list = predicted_ridge, true = y_test_data, p_last = predicted_last, y_last = y_test_data, answer = answer, x = x)

def division(a, b):
    
    if a > b:
        answer = "low"
    elif a < b:
        answer = "up"
    else:
        answer = "equal"
    return answer

@get("/prediction")
def predicting():
        
    predict_list = []
    selection = request.query.getall("model")
    
    d_ = ".\data"
    m_ = ".\model"

    d = date.today()
    str_d = str(d)

    y_test_data_file_name = str_d + "_y_test_data"
    X_test_data_file_name = str_d + "_X_test_data"

    x_test_data = np.load(os.path.join(d_, X_test_data_file_name+".npy"))
    y_test_data = np.load(os.path.join(d_, y_test_data_file_name+".npy"))
    
    x = np.arange(0, y_test_data.shape[0])
    x = x.tolist()
    json_data = {"x":x}
    fw = open(os.path.join(d_, 'x.json'), "w")
    json.dump(json_data, fw) 
    datas = []
    next_stock = []
    for model in selection:
        
        if model is "1":
            
            model = joblib.load(os.path.join(m_,"ridge.pkl"))
            predicted_ridge = model.predict(x_test_data)
            #predict_list.append(predicted_ridge)
            predicted_ridge = predicted_ridge.tolist()

            json_data = {"ridge_data":predicted_ridge}

            fw = open(os.path.join(d_, 'ridge.json'), "w")
            json.dump(json_data, fw)

            datas.append(predicted_ridge)
            next_stock.append("Ridge Regression : " + str(predicted_ridge[-1]) + "\n")
        if model is "2":
            model = joblib.load(os.path.join(m_,"lasso.pkl"))
            predicted_lasso = model.predict(x_test_data)
            #predict_list.append(predicted_lasso)
            predicted_lasso = predicted_lasso.tolist()
            fw = open(os.path.join(d_,'lasso.json'), "w")
            json.dump(predicted_lasso, fw)
            datas.append(predicted_lasso)
            next_stock.append("Lasso Regression : " + predicted_lasso[-1] + "\n")
        if model is "3":
            model = joblib.load(os.path.join(m_,"linear.pkl"))
            predicted_linear = model.predict(x_test_data)
            #predict_list.append(predicted_linear)
            predicted_linear = predicted_linear.tolist()
            fw = open(os.path.join(d_,'linear.json'), "w")
            json.dump(predicted_linear, fw)
            datas.append(predicted_linear)
            next_stock.append("linear Regression : " + predicted_linear[-1])
        if model is "4":
            model = joblib.load(os.path.join(m_,"svr.pkl"))
            predicted_svr = model.predict(x_test_data)
            #predict_list.append(predicted_svr)
            predicted_svr = predicted_svr.tolist()
            fw = open(os.path.join(d_,'svr.json'), "w")
            json.dump(predicted_svr, fw)
            datas.append(predicted_svr)
            next_stock.append("SVR Regression :  " + predicted_svr[-1] + "\n")
        if model is "5":
            json_string = open(os.path.join(m_, 'fnn_model.json')).read()
            model = model_from_json(json_string)
            model.load_weights(os.path.join(m_, 'fnn_model_weights.hdf5'))           
            predicted_fnn = model.predict(x_test_data)
            #predict_list.append(predicted_fnn)
            predicted_fnn = predicted_fnn.tolist()
            fw = open(os.path.join(d_,'fnn.json'), "w")
            json.dump(predicted_fnn, fw)
            datas.append(predicted_fnn)
            next_stock.append(predicted_fnn[-1])
        if model is "6":
            rnn_test_x = x_test_data.reshape(x_test_data.shape[0],1,x_test_data.shape[1])
            json_string = open(os.path.join(m_, 'lstm_model.json')).read()
            model = model_from_json(json_string)
            model.load_weights(os.path.join(m_, 'lstm_model_weights.hdf5'))           
            predicted_lstm = model.predict(rnn_test_x)
            #predict_list.append(predicted_rnn)      
            predicted_lstm = predicted_lstm.tolist()
            fw = open(os.path.join(d_,'lstm.json'), "w")
            json.dump(predicted_lstm, fw)
            datas.append(predicted_lstm)
            next_stock.append(predicted_lstm[-1])
    
    return template("plot", list = datas, next_stock = next_stock) 



run(host="localhost", port=8090) #ポート8080番でwebサーバーを立てる