# flask_app/server.py

# import libraries
print('importing libraries...')
from flask import Flask, request, jsonify
import logging
import random
import urllib.request
import io
import time
from torch import cuda
from torchvision import models as tmodels

from PIL import Image as PilImage
import requests, os
from io import BytesIO

# import fastai stuff
from fastai import *
from fastai.vision import *
import fastai
from fastai.imports import *
# import settings
from settings import * # importx
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
path_to_model = os.path.join(data_dir, 'models', 'model.pt')
if not os.path.exists(path_to_model):
    print('done!\nmodel weights were not found, downloading them...')
    os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    filename = Path(path_to_model)
    r = requests.get(MODEL_URL)
    filename.write_bytes(r.content)
print('done!\nloading up the saved model weights...')
defaults.device = torch.device('cpu') # run inference on cpu
#empty_data = ImageDataBunch.single_from_classes(
#    path, labels, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
#learn = create_cnn(empty_data, models.vgg16_bn)
#learn = learn.load('model')
model = tmodels.vgg16(pretrained=True)
n_classes = 5
for param in model.parameters():
    param.requires_grad = False #see my AI HW last part for an explanation of fine-tuning
    

n_inputs = model.classifier[6].in_features #get number of inputs on the layer
print(n_inputs)
# Add on classifier
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
model.idx_to_class = {
    0: "drink",
    1: "food",
    2: "inside",
    3: "menu",
    4: "outside",
}

device = torch.device('cpu')
model.load_state_dict(torch.load(path_to_model, map_location=device))
#model.load_state_dict(torch.load(path_to_model, map_location = 'cpu'))
#model = nn.DataParallel(model)
print('done!\nlaunching the server...')
def process_image(image):
    """Process an image path into a PyTorch tensor"""

   
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor

def predictImage(image, model, topk=1):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    

    # Convert to pytorch tensor
    img_tensor = process_image(image)
    train_on_gpu = cuda.is_available()
    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0][0]

        return top_p, top_classes


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
    #img = open_image(BytesIO(response.content))

    with urllib.request.urlopen(url) as url:
        f = io.BytesIO(url.read())
    img = PilImage.open(f)

    t = time.time() # get execution time
    #pred_class, pred_idx, outputs = learn.predict(img)
    #img = process_image(img)
    top_p, top_classes = predictImage(img, model)
    top_p = top_p.tolist()
    #print(top_classes)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    #app.logger.info("Image %s classified as %s" % (url, top_classes))
    display = 'Class: %s' % top_classes
    out = 'Score: %0.02f' % top_p
    return jsonify("" + display + " " + out)
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
