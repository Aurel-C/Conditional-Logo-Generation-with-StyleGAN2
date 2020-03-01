from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np

import time
import PIL.Image
from stylegan_two import StyleGAN, nImage, noise

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
latent_size = 16

@app.before_first_request
def load_model_to_app():
    app.predictor = StyleGAN()
    app.predictor.load(5)


@app.route('/', methods=['POST', 'GET'])
def index():
    img_size =32
    nSliders = latent_size
    values = {}

    for i in range(nSliders):
        values["slider"+str(i)] = 0

    if request.method == 'POST':
        values = request.form
        x = (np.array(list(values.values()), dtype="float32")).reshape((1, latent_size))
    else:
        x = np.zeros((1, latent_size))
    app.predictor.generateImage(x,np.ones((1, 32, 32, 1)) * 0.5)
    return render_template('index.html', nSliders=nSliders, val=values, size = img_size)


if __name__ == "__main__":
    app.run(debug=True)

