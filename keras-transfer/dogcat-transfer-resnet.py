# modified from https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

# path to the model weights files.
#weights_path = '../keras/examples/vgg16_weights.h5'
#top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'dogcat-data/train'
validation_data_dir = 'dogcat-data/validation'
#nb_train_samples = 2000
#nb_validation_samples = 2000
epochs = 50
batch_size = 10
output_classes = 2

inp = Input(shape=(224, 224, 3), name='input_image')

main_model = applications.ResNet50(include_top=False)
for layer in main_model.layers:
    layer.trainable=False

main_model = main_model(inp)
main_out = Flatten()(main_model)
main_out = Dense(512, activation='relu', name='fcc_0')(main_out)
main_out = Dense(1, activation='softmax', name='class_id')(main_out)

model = Model(input=inp, output=main_out)


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

print(train_generator.classes)

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=100,#nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[WandbKerasCallback()],

    nb_val_samples=40
    )
