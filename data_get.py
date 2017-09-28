
from bs4 import BeautifulSoup
import numpy as np
import urllib.request
import re
import os
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,LSTM
from keras import regularizers
from datetime import date

def remove_td(data):
    clean_data = []
    for d in data:
        cd = re.sub(r'<?td>','', d)
        clean_data.append(re.sub(r',','', cd))
    clean_data = list(map(int, clean_data))
    return clean_data

def make_url():
    urls = []
    url_ = "https://info.finance.yahoo.co.jp/history/?code=6193.T&sy=2016&sm=1&sd=1&ey=2017&em=9&ed=21&tm=d&p="
    _url = [str(i) for i in range(1,20)]
    for _ in _url:
        url = url_ + _
        urls.append(url)
    return urls

def make_data():
    urls = make_url()       
    sum_first = []
    sum_top = []
    sum_tail = []
    sum_last = []
    
    for url in urls[::-1]:
        pattern = r"\d{4}年\d{1,2}月\d{1,2}日"
        print(url)
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        html = response.read()
        soup = BeautifulSoup(html, "lxml")
        td_data = soup.find_all('td')
    
        first = []
        top = []
        tail = []
        last = []

        repattern = re.compile(pattern)
    
        for (i, data) in enumerate (td_data):
            m = repattern.match(data.text)
            if m:
                first.append(td_data[i+1].text)
                top.append(td_data[i+2].text)
                tail.append(td_data[i+3].text)
                last.append(td_data[i+4].text)
        
        first = [i for i in first[::-1]]
        top = [i for i in top[::-1]]
        tail = [i for i in tail[::-1]]
        last = [i for i in last[::-1]]
        
        first = remove_td(first)
        top = remove_td(top)
        tail = remove_td(tail)
        last = remove_td(last)
                
        sum_first.append(first)
        sum_top.append(top)
        sum_tail.append(tail)
        sum_last.append(last)

    return sum_first, sum_top, sum_tail, sum_last

def convert_array_concat( sum_data):
    sum_data_arr = np.array(sum_data)
    f = sum_data_arr[0]
    for i in range(1, sum_data_arr.shape[0]):
        f = np.append(f,sum_data_arr[i])
    return f

def make_y_data(sum_tail):
    
    yy = []
    y = sum_tail
    last = y[-1]
    first = y[0]
    yy.append([y[i - 1] for i in range(0, y.shape[0]-1)])
    yy = np.append(yy, last)

    return yy

def stack_data(sum_first, sum_top, sum_tail, sum_last):
    
    _1 =  np.vstack((sum_first, sum_top))
    _2 = np.vstack((_1, sum_tail))
    _3 = np.vstack((_2, sum_last))
    transported_matrix = _3.T
    
    return transported_matrix

def ridge_regressoin(train_data_x, train_data_y,m_):
    print("Training ridge regression")
    model = linear_model.Ridge()
    model.fit(train_data_x, train_data_y)

    joblib.dump(model, os.path.join(m_,"ridge.pkl"))

def lasso_regressoin(train_data_x, train_data_y ,m_):
    print("Training lasso regression")
    model = linear_model.Lasso(alpha = 0.0001)
    model.fit(train_data_x, train_data_y)
    
    joblib.dump(model, os.path.join(m_,"lasso.pa ==kl"))
    
def linear_regression(train_data_x, train_data_y ,m_):
    print("Training linear regression")
    model = linear_model.LinearRegression()
    model.fit(train_data_x, train_data_y)

    joblib.dump(model, os.path.join(m_,"linear.pkl"))

def support_vecter_regression(train_data_x, train_data_y ,m_):
    print("Training Support Vector Regression")
    tuned_parameters = [
    {'C': [0.01, 0.1,1, 10], 'kernel': ['linear']},
    {'C': [0.01, 0.1,1, 10], 'kernel': ['rbf'], 'gamma': [0.01,0.001, 0.0001]},
    ]

    clf_svr = GridSearchCV(svm.SVR(),
                    tuned_parameters,
                    cv = 5
                    )
    clf_svr.fit(train_data_x, train_data_y)

    joblib.dump(clf_svr, os.path.join(m_,"svr.pkl"))

def feed_forward_neural_network(train_data_x, train_data_y ,m_):
    print("Training FNN")
    fnn = Sequential()
    fnn.add(Dense(512, input_dim=4))
    fnn.add(Activation('relu'))
    fnn.add(Dropout(0.5))
    fnn.add(Dense(1, input_dim=512))
    fnn.add(Activation('linear'))
    fnn.compile(optimizer="adam",
        loss = "mse")
    fnn.fit(train_data_x, train_data_y, epochs=100, batch_size=70, validation_split=0.1)
    json_string = fnn.to_json()
    open(os.path.join(m_,'fnn_model.json'), 'w').write(json_string)
    fnn.save_weights(os.path.join(m_,'fnn_model_weights.hdf5'))
    
def long_short_term_memory(train_data_x, train_data_y ,m_):
    print("Training LSTM")
    rnn_train_x = train_data_x
    rnn_test_x = test_data_x   
    rnn_train_x = rnn_train_x.reshape(train_data_x.shape[0],1,train_data_x.shape[1])
    rnn_test_x = rnn_test_x.reshape(test_data_x.shape[0],1,test_data_x.shape[1])
    lstm = Sequential()
    lstm.add(LSTM(516, input_dim=4, input_length = 1, return_sequences = True))
    lstm.add(Activation("relu"))
    lstm.add(Dropout(0.5))
    lstm.add(LSTM(1, input_dim = 516, input_length = 1))
    lstm.add(Activation("linear"))
    lstm.compile(optimizer="adam",
        loss = "mse")
    lstm.fit(rnn_train_x, train_data_y, epochs=15, batch_size=7, validation_split=0.1)
    json_string = lstm.to_json()
    open(os.path.join(m_,'lstm_model.json'),'w').write(json_string)
    lstm.save_weights(os.path.join(m_,'lstm_model_weights.hdf5'))

  
if __name__ == '__main__':
    
    d_ = ".\data"
    m_ = ".\model"

    d = date.today()
    str_d = str(d)

    y_data_file_name = str_d + "_y_data"
    X_data_file_name = str_d + "_X_data"

    y_test_data_file_name = str_d + "_y_test_data"
    X_test_data_file_name = str_d + "_X_test_data"

    try:
        y_data = np.load(os.path.join(d_, y_data_file_name+".npy"))
        X_data = np.load(os.path.join(d_, X_data_file_name+".npy"))    
    
    except:
        sum_first, sum_top, sum_tail, sum_last = make_data()
        s_list = [sum_first, sum_top, sum_tail, sum_last]
        sum_first = convert_array_concat(sum_first)
        sum_top = convert_array_concat(sum_top)
        sum_tail = convert_array_concat(sum_tail)
        sum_last = convert_array_concat(sum_last)
        
        y_data = make_y_data(sum_tail)
        X_data = stack_data(sum_first, sum_top, sum_tail, sum_last)

          
        np.save(os.path.join(d_,y_data_file_name), y_data)
        np.save(os.path.join(d_,X_data_file_name), X_data)

    finally:
        
        X_scaled = preprocessing.scale(X_data)
        X_scaled.std(axis=0)

        split_num = 0.2
        ratio = int(X_scaled.shape[0] * split_num)
        train_data_x = X_scaled[:][ratio:]
        test_data_x = X_scaled[:][:ratio]
        yy = make_y_data(sum_tail)
        train_data_y = yy[ratio:]
        test_data_y = yy[:ratio]
            
        np.save(os.path.join(d_,y_test_data_file_name), test_data_y)
        np.save(os.path.join(d_,X_test_data_file_name), test_data_x)

          #y_classify = [1  if sum_last[i + 1] > sum_last[i] else 0 for i in range(sum_last.shape[0] - 1)]
        #if sum_last[-1] > sum_last[0]:
        #    y_classify = np.append(y_classify, 1)
            #else:
         #    y_classify = np.append(y_classify, 0)
        #train_data_y = y_classify[ratio:]
        #test_data_y = y_classify[:ratio]
        train_data_y.reshape(-1)
        test_data_y.reshape(-1)

        ridge_regressoin(train_data_x, train_data_y,m_)
        lasso_regressoin(train_data_x, train_data_y, m_)
        linear_regression(train_data_x, train_data_y, m_)
        support_vecter_regression(train_data_x, train_data_y, m_)
        feed_forward_neural_network(train_data_x, train_data_y, m_)
        long_short_term_memory(train_data_x, train_data_y, m_)
        




