# flask_app/server.py

# import libraries
print('importing libraries...')
from flask import Flask, request, jsonify
import logging
import random
import time

from PIL import Image
import requests, os
from io import BytesIO

# import fastai stuff
from fastai import *
from fastai.vision import *
import fastai
from fastai.imports import *
# import settings
from settings import * # import
import torch
print('done!\nsetting up the directories and the model structure...')
# set dir structure
def make_dirs(labels, data_dir):
    root_dir = os.getcwd()
    make_dirs = ['train', 'valid', 'test']
    for n in make_dirs:
        name = os.path.join(root_dir, data_dir, n)
        for each in labels:
            os.makedirs(os.path.join(name, each), exist_ok=True)
make_dirs(labels=labels, data_dir=data_dir) # comes from settings.py
path = Path(data_dir)

# download model weights if not already saved
path_to_model = os.path.join(data_dir, 'models', 'model.pth')
if not os.path.exists(path_to_model):
    print('done!\nmodel weights were not found, downloading them...')
    os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    filename = Path(path_to_model)
    r = requests.get(MODEL_URL)
    filename.write_bytes(r.content)
print('done!\nloading up the saved model weights...')
defaults.device = torch.device('cpu') # run inference on cpu
empty_data = ImageDataBunch.single_from_classes(
    path, labels, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(empty_data, models.resnet34)
learn = learn.load('model')
print('done!\nlaunching the server...')
# set flask params
app = Flask(__name__)
@app.route("/")
def hello():
    return "Image classification example\n"
@app.route('/predict', methods=['GET'])
def predict():
    url = request.args['url']
    app.logger.info("Classifying image %s" % (url),)

    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    t = time.time() # get execution time
    pred_class, pred_idx, outputs = learn.predict(img)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    app.logger.info("Image %s classified as %s" % (url, pred_class))
    return jsonify(pred_class)
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
