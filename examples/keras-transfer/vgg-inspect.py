from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import numpy as np

if __name__ == "__main__":
    model = VGG16()
    model.summary()
    im = cv2.resize(cv2.imread('elephant.jpg'),
                    (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    out = model.predict(im)
    print(np.argmax(out))
