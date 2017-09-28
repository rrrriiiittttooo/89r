from bottle import get, run, template, request, post
import subprocess
import data_get
#import matplotlib.pyplot as plt

@get("/")
def home():
    return template("index")

@get("/start")
def home():
    command = "python data_get.py"
    subprocess.call(command)
    return template("index")


@get("/prediction")
def predict():
    #model_names = []
    #model_names = request.query.get("a")
    #d_ = "./data"
    #m_ = "./model"
    #np.load(os.path.join(d_,"y_test_data"))
    #np.load(os.path.join(d_,"X_test_data"))]
    return


@get("/prediction")
def predict():
    d_ = "./data"
    m_ = "./model"
    command = "python data_get.py"
    subprocess.call(command)
    #np.load(os.path.join(d_,"y_test_data"))
    #np.load(os.path.join(d_,"X_test_data"))]
    return
