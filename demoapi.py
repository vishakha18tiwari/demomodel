from __future__ import print_function
import numpy as np
from flask import Flask, jsonify, render_template,  request

import pickle
import codecs, json
import sys
my_random_forest=pickle.load(open("iris_rfc.pkl","rb"))
# initializing a variable of Flask
app=Flask(__name__,template_folder='template')
# decorating index function with the app.route with url as /login
@app.route('/login')
def index():
    return render_template('login.html')

@app.route('/FlaskTutorial', methods=['POST','GET'])
def success():
    sl=request.form['sl']
    sw=request.form['sw']
    pl=request.form['pl']
    pw=request.form['pw']
    predict_request=[sl,sw,pl,pw]
    predict_request=np.asarray(predict_request)
    result=int(my_random_forest.predict(predict_request)[0])
    return render_template('success.html',result=result)
app.run(port=5000)

