# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/json', methods=['POST'])
def predict():
    if TD:
        try:
            json_ = request.get_json()
            query = pd.get_dummies(pd.DataFrame(json_))
            prediction = list(TD.predict(query))
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
@app.route('/archivo', methods=['POST'])
def predict2():
    if TD:
        try:
            json_ = pd.read_json('temp.json', orient='records', lines=True)
            query = pd.get_dummies(pd.DataFrame(json_))
            prediction = list(TD.predict(query))
            return jsonify({'prediction': str(prediction)})
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
if __name__ == '__main__':
    TD = joblib.load("model2.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns3.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(debug=True)