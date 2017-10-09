# need to do pip install pillow to run

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

model = InceptionV3(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
model.summary()
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
model.save('image.h5')
