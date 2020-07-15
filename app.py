from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from stylegan_two import StyleGAN, nImage, noise

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
latent_size = 16
app.predictor = StyleGAN()
app.predictor.load(10)
# # Chargement du modele
# @app.before_first_request
# def load_model_to_app():


#Chargement de la page
@app.route('/', methods=['POST', 'GET'])
def index():
    img_size =128
    nSliders = latent_size
    if request.method == 'POST':
        values = request.form
    else:
        values = {}
        values["label"] = 0
        values["noise"] = 0.5
        for i in range(nSliders):
            values["slider"+str(i)] = 0
            values["style"+str(i)] = 0

    label = int(values.get("label"))
    x = (np.array([v for k,v in values.items() if 'slider' in k], dtype="float32")).reshape((1, latent_size))
    style = (np.array([v for k,v in values.items() if 'style' in k], dtype="float32")).reshape((1, latent_size))
    noise = float(values.get("noise"))
    app.predictor.generateImage(x,label,np.ones((1, img_size, img_size, 1))*noise,style)
    return render_template('index.html', nSliders=nSliders, val=values,size = img_size)


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)

