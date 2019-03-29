import flask
import keras
import numpy as np
import os
from keras.models import load_model
from PIL import Image
from flask import Flask, request
from jinja2 import Template

app = Flask(__name__)

model = load_model('smile.h5')
model._make_predict_function()


def predict_image(image):
    image = image.convert(mode="L")
    image = image.resize((64, 64))
    im = np.asarray(image)
    im_rescale = im.reshape(1, 64, 64, 1)
    pred = model.predict(im_rescale)
    return pred[0]


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files['file']
    image = Image.open(f.stream)
    pred = predict_image(image)
    template = Template("""
        <html>
            <body>
                <p>Probability of Smiling: {{smile_prob}}</p>
                <p>Probability of Not Smiling: {{no_smile_prob}}</p>
            </body>
        </html>
    """)

    return template.render(smile_prob=pred[0], no_smile_prob=pred[1])


@app.route("/")
def index():
    html = """
    <html>
        <body>
            <form action="predict" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*;capture=camera">
                <input type="submit"/>
            </form>
        </body>
    </html>
    """
    return(html)


if __name__ == '__main__' and not os.getenv("FLASK_DEBUG"):
    app.run(port=8080)
