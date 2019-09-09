# modified from https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
import glob
import wandb
from wandb.keras import WandbCallback
from dogcat_data import generators, get_nb_files

run = wandb.init()
config = run.config
config.img_width = 224 
config.img_height = 224
config.epochs = 50
config.batch_size = 32

inp = Input(shape=(224, 224, 3), name='input_image')

main_model = ResNet50(include_top=False, weights="imagenet")
for layer in main_model.layers:
    layer.trainable=False

main_model = main_model(inp)
main_out = Flatten()(main_model)
main_out = Dense(512, activation='relu', name='fcc_0')(main_out)
main_out = Dense(1, activation='sigmoid', name='class_id')(main_out)

model = Model(inputs=inp, outputs=main_out)
model._is_graph_network = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_dir = "dogcat-data/train"
val_dir = "dogcat-data/validation"
nb_train_samples = get_nb_files(train_dir)
nb_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(val_dir)

train_generator, validation_generator = generators(preprocess_input, config.img_width, config.img_height, config.batch_size, binary=True)

# fine-tune the model
model.fit_generator(
    train_generator,
    epochs=config.epochs,
    validation_data=validation_generator,
    callbacks=[WandbCallback(data_type="image", generator=validation_generator, labels=['cat', 'dog'],save_model=False)],
    workers=2,
    steps_per_epoch=nb_train_samples * 2 / config.batch_size,
    validation_steps=nb_train_samples / config.batch_size,
)
model.save('transfered.h5')
