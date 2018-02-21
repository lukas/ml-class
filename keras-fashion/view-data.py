import keras
from keras.datasets import fashion_mnist
from PIL import Image
from PIL import ImageDraw 

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

for i in range(10):
    img = Image.fromarray(X_train[i])
    img = img.resize((280, 280), Image.ANTIALIAS)

    draw = ImageDraw.Draw(img)
    draw.text((10, 10),labels[y_train[i]],(255))
    img.save(str(i)+".jpg")