from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from stylegan_two import StyleGAN, nImage, noise

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
img_size = 128
latent_size = 64
n_images = 4
# Loading model
app.predictor = StyleGAN()
app.predictor.load(12)

# Creating latents inputs
app.base_latent = np.random.normal(size=(1,latent_size))
app.latents = np.random.normal(loc=app.base_latent,scale=0.2,size=(n_images,1,latent_size))
# Creating styles inputs
app.base_style = np.random.normal(size=(1,latent_size))
app.styles = np.random.normal(loc=app.base_style,scale=0.5,size=(n_images,1,latent_size))

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        values = request.form
        if values.get("pattern") is not None:
            if values.get("pattern")=="rand":
                app.latents = np.random.normal(size=(n_images,1,latent_size))
            elif values.get("pattern")=="autre":
                app.latents = np.random.normal(loc=app.base_latent,scale=0.2,size=(n_images,1,latent_size))
            elif int(values.get("pattern")) in range(n_images):
                app.base_latent = app.latents[int(values.get("pattern"))]
                app.latents = np.random.normal(loc=app.base_latent,scale=0.2,size=(n_images,1,latent_size))
        elif values.get("color") is not None:
            if values.get("color")=="rand":
                app.styles = np.random.normal(size=(n_images,1,latent_size))
            elif values.get("color")=="autre":
                app.styles = np.random.normal(loc=app.base_latent,scale=0.5,size=(n_images,1,latent_size))
            elif int(values.get("color")) in range(n_images):
                app.base_style = app.styles[int(values.get("color"))]
                app.styles = np.random.normal(loc=app.base_style,scale=0.5,size=(n_images,1,latent_size))
    else:
        values = {}
        values["label"] = 0
        values["noise"] = 0.5

    label = int(values.get("label"))
    noise = float(values.get("noise"))

    app.predictor.generateImage(app.base_latent,label,np.ones((1, img_size, img_size, 1))*noise,app.base_style,"static/image/image.jpg")
    for i in range(app.latents.shape[0]):
        app.predictor.generateImage(app.latents[i],label,np.ones((1, img_size, img_size, 1))*noise,app.base_style,"static/image/logo/pattern/i"+str(i)+".jpg")
    for i in range(app.styles.shape[0]):
        app.predictor.generateImage(app.base_latent,label,np.ones((1, img_size, img_size, 1))*noise,app.styles[i],"static/image/logo/color/i"+str(i)+".jpg")
    return render_template('index.html', val=values)


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)

