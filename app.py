from os import pipe
from IPython import display
from flask import Flask, render_template, request
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open("diabetes_model.pkl","rb")) 

@app.route('/')
def index(): 
    return render_template('UI.html')

@app.route('/predict', methods =['POST'])
def predict():
    age= float(request.form.get("age"))
    gender=float(request.form.get("gender"))
    bmi=float(request.form.get("bmi"))
    bp=float(request.form.get("bp"))
    S1=float(request.form.get("S1"))
    S2=float(request.form.get("S2"))
    S3=float(request.form.get('S3'))
    S4=float(request.form.get("S4"))
    S5=float(request.form.get("S5"))
    S6=float(request.form.get("S6"))

    input = pd.DataFrame([[age, gender, bmi, bp, S1, S2, S3, S4, S5, S6]],
                    columns=["age", "gender", "bmi", "bp", "S1", "S2", "S3", "S4", "S5", "S6"])
 
    prediction = model.predict(input)[0]

    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
