from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
import keras
from keras.models import load_model
import keras.backend as K

# very important to do this as a first thing
K.set_learning_phase(0)

model = load_model('smile.h5')


export_path = 'tensorflow'
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'images': model.input},
                                  outputs={'scores': model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()
    print("Saved")
