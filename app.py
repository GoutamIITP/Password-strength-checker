from flask import Flask, redirect,url_for,render_template,request
import pickle
import numpy as np
from utility import load_pass_model, predict_password_strength,extract_features

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

 
@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == "POST":
        exampleInputPassword1 = request.form['password']
        
        pass_model = load_pass_model()
        prediction = predict_password_strength(exampleInputPassword1, pass_model)
        return render_template("index.html", strength=prediction)
    else:
        return render_template('index.html', strength="")
              
    
if __name__=='__main__':
    app.run(debug=True)
