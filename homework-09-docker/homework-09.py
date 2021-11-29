#!/usr/bin/env python
# coding: utf-8

from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np
import tflite_runtime.interpreter as tflite

def load_model():
    MODEL_PATH = 'cats-dogs-v2.tflite'
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    return interpreter


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


INTERPRETER = load_model()
INPUT_INDEX = INTERPRETER.get_input_details()[0]['index']
OUTPUT_INDEX = INTERPRETER.get_output_details()[0]['index']
INPUT_SHAPE = (150, 150, 3)


def predict(url):
    img = prepare_image(download_image(url), INPUT_SHAPE[:2])
    img_array = np.asarray(img).astype(np.float32) / 255

    INTERPRETER.set_tensor(INPUT_INDEX, [img_array])
    INTERPRETER.invoke()

    return INTERPRETER.get_tensor(OUTPUT_INDEX)[0]


if __name__ == '__main__':
    url = 'https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg'
    print(predict(url))
