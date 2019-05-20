import flask
import keras
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ExifTags
from flask import Flask, request
from jinja2 import Template
import base64

app = Flask(__name__)

model = load_model('smile.h5')
model._make_predict_function()


def maybe_rotate(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        return image


def predict_image(image):
    image = maybe_rotate(image)
    image = image.convert(mode="L")
    image = image.resize((64, 64))
    im = np.asarray(image)
    im = im.reshape(1, 64, 64, 1)

    im_rescale = im / 255.0
    pred = model.predict(im_rescale)
    return pred[0], image


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files['file']
    image = Image.open(f.stream)
    pred, image = predict_image(image)
    image.save("/tmp/thumb_file.jpg")
    with open("/tmp/thumb_file.jpg", "rb") as img:
        thumb_string = base64.b64encode(img.read())
        base64out = "data:image/jpeg;base64," + str(thumb_string)[2:-1]
    template = Template("""
        <html>
            <body>
                <img src="{{face}}" style="width:200px" />
                <p>Probability of Smiling: {{smile_prob}}</p>
                <p>Probability of Not Smiling: {{no_smile_prob}}</p>
            </body>
        </html>
    """)

    return template.render(smile_prob="%.2f" % pred[1], no_smile_prob="%.2f" % pred[0], face=base64out)


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
