from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from waitress import serve
import imageio as iio
import numpy as np
import tensorflow as tf

ESTIMATOR = load_model('model.h5')
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_price():
    flat = request.get_data().decode('utf-8')
    return jsonify(predict(flat))


def predict(url):
    img = iio.imread(url).reshape(-1, 48, 48, 3)
    prediction = ESTIMATOR.predict(tf.convert_to_tensor(img))[0]
    expression = LABELS[np.argmax(prediction)]
    return {'expression': expression}


if __name__ == '__main__':
    host_port = '0.0.0.0:8080'
    print(f'Listening at {host_port}')
    serve(app, listen=host_port)
