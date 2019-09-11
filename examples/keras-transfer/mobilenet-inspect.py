from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import time

model = MobileNetV2(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
model.summary()
t = time.time()
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
print('Took: %.2f seconds' % (time.time() - t))
# model.save('image.h5')
