from IPython.display import display
import jsonify
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open("diabetes_model.pkl","rb")) 

@app.route('/')
def upload():
    return render_template("UI.html")

@app.route('/batch_predict', methods =['POST'])
def batch_predict(): 
    file = request.files['file']  
    df = pd.read_csv(file) 
    pred = model.predict(df)
    df_pred  = pd.DataFrame(pred, columns=['Prediction'])
    df['Prediction'] = df_pred
    html = df.to_html()
    batch_prediction = html
    return batch_prediction


if __name__ == '__main__':
    app.run(debug=True, port = 5050)
