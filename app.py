import pickle
import numpy as np
import json
import os
from flask import Flask, redirect, render_template, request
import requests
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
res, data = '', []


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global res, data
    data = request.form
    data = list(data.values())
    datas = np.array([int(i) for i in data[1:]]).reshape(1, -1)
    model = pickle.load(open('model.sav', 'rb'))
    res = model.predict(datas)
    if request.method == 'POST':
        return redirect("/report")
    else:
        return redirect("/")


@app.route('/report', methods=['GET', 'POST'])
def report():
    if res:
        jsonCont = requests.get(
            'https://www.jsonkeeper.com/b/'+os.environ.get("SECRET"))
        jsonCont = jsonCont.json()
        return render_template('success.html', res=res[0], ptr=jsonCont[res[0]], data=json.dumps(data))
    return redirect("/")
