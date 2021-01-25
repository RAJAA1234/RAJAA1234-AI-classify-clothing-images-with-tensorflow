from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Image loader
import cv2

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28))
    img_array = np.asarray(img)
    x = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    result = int(img_array[0][0][0])
    print(result)
    if result > 128:
      img = cv2.bitwise_not(x)
    else:
      img = x
    img = img/255
    img = (np.expand_dims(img,0))

    preds =  model.predict(img)
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

