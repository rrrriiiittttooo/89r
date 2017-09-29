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

@route('/image/<filename>')
def css_dir(filename):
    return static_file(filename, root=BASE_DIR + "\\views\static\image\\")

@get("/")
def hello():
    return template("free")

@get("/training/model")
def training():
    d_ = "./data"
    m_ = "./model"
    command = "python data_get.py"
    subprocess.call(command)
    return template("menu")

@get("/menu")
def menu():
    return template("menu")

@get('/ml/<modelname>')
def predict(modelname):
    
    d_ = ".\data"
    m_ = ".\model"

    model_file_name = modelname + ".pkl"
    print(model_file_name)
    data_file_name = "predicted_" + modelname    
    d = date.today()
    str_d = str(d)
    
    y_test_data_file_name = str_d + "_y_test_data"
    X_test_data_file_name = str_d + "_X_test_data"

    x_test_data = np.load(os.path.join(d_, X_test_data_file_name+".npy"))
    y_test_data = np.load(os.path.join(d_, y_test_data_file_name+".npy"))
    x = np.arange(0, y_test_data.shape[0]-1)
    x = x.tolist()
    y_test_data = y_test_data.tolist()
    
    model = joblib.load(os.path.join(m_,model_file_name))
    data_file_name = model.predict(x_test_data)
    data_file_name = data_file_name.tolist()
    answer = high_Or_Low(y_test_data[-2], data_file_name[-1])
    predicted_last = int(data_file_name[-1])
    y_last = int(y_test_data[-2])#正解の値
    data_file_name = data_file_name[:-1]
    y_test_data = y_test_data[:-1]
    
    return template("plots", list =data_file_name, true = y_test_data, p_last = predicted_last, y_last = y_last, answer = answer, x = x)

def high_Or_Low(a, b):
    
    if a > b:
        answer = "low"
    elif a < b:
        answer = "up"
    else:
        answer = "equal"
    return answer

@get('/dnn/fnn')
def predict():
    
    d_ = ".\data"
    m_ = ".\model"
    
    d = date.today()
    str_d = str(d)
    
    X_test_data_file_name = "DNN_" + str_d + "_X_test_data.npy"
    y_test_data_file_name = str_d + "_y_test_data.npy"

    x_test_data = np.load(os.path.join(d_, X_test_data_file_name))
    y_test_data = np.load(os.path.join(d_, y_test_data_file_name))
    x = np.arange(0, y_test_data.shape[0]-1)
    x = x.tolist()
    json_string = open(os.path.join(m_, 'fnn_model.json')).read()
    model = model_from_json(json_string)
    model.load_weights(os.path.join(m_, 'fnn_model_weights.hdf5'))          
    predicted_fnn = model.predict(x_test_data).flatten().tolist()
    y_test_data = y_test_data.tolist()
    answer = high_Or_Low(y_test_data[-2], predicted_fnn[-1])
    predicted_last = int(predicted_fnn[-1])
    y_last = int(y_test_data[-2])
    predicted_fnn = predicted_fnn[:-1]
    y_test_data = y_test_data[:-1]
    
    return template("plots", list =predicted_fnn, true = y_test_data, p_last = predicted_last, y_last = y_last, answer = answer, x = x)

def high_Or_Low(a, b):
    a = float(a)
    b = float(b)
    if a > b:
        answer = "Down"
    elif a < b:
        answer = "Up"
    else:
        answer = "equal"
    return answer

@get('/dnn/rnn')
def predict():
    print("IN!")
    d_ = ".\data"
    m_ = ".\model"
    
    d = date.today()
    str_d = str(d)
    
    X_test_data_file_name = "DNN_" + str_d + "_X_test_data.npy"
    y_test_data_file_name = str_d + "_y_test_data.npy"
        
    x_test_data = np.load(os.path.join(d_, X_test_data_file_name))
    y_test_data = np.load(os.path.join(d_, y_test_data_file_name))
    
    rnn_test_x = x_test_data.reshape(x_test_data.shape[0],1,x_test_data.shape[1])
    
    x = np.arange(0, y_test_data.shape[0]-1)
    x = x.tolist()
    y_test_data = y_test_data.tolist()
    
    json_string = open(os.path.join(m_, 'rnn_model.json')).read()
    model = model_from_json(json_string)
    model.load_weights(os.path.join(m_, 'rnn_model_weights.hdf5'))           
    predicted_rnn = model.predict(rnn_test_x).flatten().tolist()
    print(predicted_rnn[-1], y_test_data[-2])
    answer = high_Or_Low2(y_test_data[-2], predicted_rnn[-1])
    predicted_last = predicted_rnn[-1]
    y_last = y_test_data[-2]
    y_test_data = y_test_data[:-1]
    
    return template("plots", list =predicted_rnn, true = y_test_data, p_last = predicted_last, y_last = y_last, answer = answer, x = x)
def high_Or_Low2(a, b):
    a = float(a)
    b = float(b)
    if a > b:
        answer = "low"
    elif a < b:
        answer = "Up"
    else:
        answer = "equal"
    return answer

run(host='0.0.0.0', port=8090, debug=True) #ポート8090番でwebサーバーを立てる


