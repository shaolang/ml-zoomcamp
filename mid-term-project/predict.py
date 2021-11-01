from flask import Flask, request, jsonify
from joblib import load
from waitress import serve
import numpy as np
import pandas as pd

ESTIMATOR = load('pipeline.bin')
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_price():
    flat = request.get_json()
    print(flat)
    return jsonify(predict(flat))


def predict(flat):
    df = pd.DataFrame([flat])
    prediction = ESTIMATOR.predict(df)[0]
    return {'estimated_price': round(np.expm1(prediction), 2)}

if __name__ == '__main__':
    host_port = '0.0.0.0:8080'
    print(f'Listening at {host_port}')
    serve(app, listen=host_port)
