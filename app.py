from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from stylegan_two import StyleGAN, nImage, noise

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
img_size = 128
latent_size = 64
n_images = 4
# Chargement du modele
app.predictor = StyleGAN()
app.predictor.load(12)
base_latent = np.random.normal(size=(1,latent_size))
latents = np.random.normal(loc=base_latent,scale=0.2,size=(n_images,1,latent_size))

base_style = np.random.normal(size=(1,latent_size))
styles = np.random.normal(loc=base_style,scale=0.5,size=(n_images,1,latent_size))

@app.route('/', methods=['POST', 'GET'])
def index():
    global base_latent
    global base_style
    global latents
    global styles
    if request.method == 'POST':
        values = request.form
        if values.get("motif") is not None:
            if values.get("motif")=="rand":
                latents = np.random.normal(size=(n_images,1,latent_size))
            elif values.get("motif")=="autre":
                latents = np.random.normal(loc=base_latent,scale=0.2,size=(n_images,1,latent_size))
            elif int(values.get("motif")) in range(n_images):
                base_latent = latents[int(values.get("motif"))]
                latents = np.random.normal(loc=base_latent,scale=0.2,size=(n_images,1,latent_size))
        elif values.get("couleur") is not None:
            if values.get("couleur")=="rand":
                styles = np.random.normal(size=(n_images,1,latent_size))
            elif values.get("couleur")=="autre":
                styles = np.random.normal(loc=base_latent,scale=0.5,size=(n_images,1,latent_size))
            elif int(values.get("couleur")) in range(n_images):
                base_style = styles[int(values.get("couleur"))]
                styles = np.random.normal(loc=base_style,scale=0.5,size=(n_images,1,latent_size))
    else:
        values = {}
        values["label"] = 0
        values["noise"] = 0.5

    label = int(values.get("label"))
    noise = float(values.get("noise"))

    app.predictor.generateImage(base_latent,label,np.ones((1, img_size, img_size, 1))*noise,base_style,"static/image/image.jpg")
    for i in range(latents.shape[0]):
        app.predictor.generateImage(latents[i],label,np.ones((1, img_size, img_size, 1))*noise,base_style,"static/image/logo/motif/i"+str(i)+".jpg")
    for i in range(styles.shape[0]):
        app.predictor.generateImage(base_latent,label,np.ones((1, img_size, img_size, 1))*noise,styles[i],"static/image/logo/couleur/i"+str(i)+".jpg")
    return render_template('index.html', val=values)


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)

