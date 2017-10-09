from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')
model = InceptionV3(weights='imagenet')
