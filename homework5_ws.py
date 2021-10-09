from flask import Flask, request, jsonify
from glob import glob
from waitress import serve
import pickle

def load_python_object_from_file(fname):
    with open(fname, 'rb') as fin:
        obj = pickle.load(fin)

    return obj

DV = load_python_object_from_file('dv.bin')
MODEL_FILE = glob('model*.bin')[0]
MODEL = load_python_object_from_file(MODEL_FILE)
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_churn():
    customer = request.get_json()
    return jsonify(predict(customer))

def predict(customer):
    prediction = MODEL.predict_proba(DV.transform(customer))[0][1]
    return {'churn_probability': float(prediction), 'churn': bool(prediction)}


if __name__ == '__main__':
    host_port = 'localhost:8080'

    print(f'Listening at {host_port}')
    serve(app, listen=host_port)
