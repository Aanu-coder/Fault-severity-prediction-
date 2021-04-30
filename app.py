# Deploying the Model ###############
#It is now time to deploy our machine learning model as a REST API

import pickle
from flask import Flask, request, json, jsonify,render_template, request
from Cart_model import filename,loaded_model
import numpy as np
import pandas as pd
 
app = Flask(__name__)
#---the filename of the saved model---
filename = 'telstrafinalized_model.sav'
 
#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))
cols = ["Location", "Severity Type", "Resource Type", "Log Feature", "Volume", "Event Type"]
 

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features =[x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final],columns=cols)
    prediction = loaded_model.predict(data_unseen)
    return render_template('home.html',pred='Fault severity is: {}'.format(prediction))

 
if __name__ == '__main__':
    app.run(debug=True)