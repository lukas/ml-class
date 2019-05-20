from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = load_model("convnet.h5")

print(model.summary())

display_width = 8
display_height = 8

def visualize_layer(out, depth):
    if len(out.shape)==4:
        for i in range(min(8,out.shape[3])):
            plt.subplot2grid((display_height, display_width), (depth,i))

            a= plt.imshow(out[0,:,:,i])
            a.set_cmap('hot')
            a.axes.get_xaxis().set_visible(False)
            a.axes.get_yaxis().set_visible(False)

    else:
        plt.subplot2grid((display_height, display_width), (depth-1,0), colspan=8)
        if (len(out[0,:]) > 10):
            a = plt.imshow(out[0,:].reshape(-1, 32))
        else:
            a = plt.imshow(out[0,:].reshape(-1, 10))

        a.set_cmap('hot')
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

def visualize(t_i):
    X_t = X_test[t_i].reshape(1,28,28,1)
    print(model.predict(X_t))

    plt.axis('off')
    fig = plt.figure(1)

    gridspec.GridSpec(display_height, display_width)

    for depth in range(0,len(model.layers)):
        test_model = Sequential()
        for layer in model.layers[0:depth+1]:
            test_model.add(layer)

        if depth == 0:
            out = X_t
        else:
            out = test_model.predict(X_t)

        visualize_layer(out, depth)


    plt.show()

visualize(100)
