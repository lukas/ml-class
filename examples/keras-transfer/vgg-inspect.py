from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

img_path = 'elephant.jpg'
model = VGG16()
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
model.summary()
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
