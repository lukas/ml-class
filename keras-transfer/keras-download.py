from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import subprocess

model = ResNet50(weights='imagenet')
model = InceptionV3(weights='imagenet')
print("Downloading dog/cat dataset...")
subprocess.check_output("curl https://storage.googleapis.com/wandb-production.appspot.com/qualcomm/dogcat-data.tgz | tar xvz", shell=True)
subprocess.check_output("rm dogcat-data/train/dog/._dog* dogcat-data/train/cat/._cat* dogcat-data/validation/cat/._cat* dogcat-data/validation/dog/._dog*", shell=True)
