import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow import keras
import numpy as np
import wandb


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def repeated_predictions(model, data, look_back, steps=100):
    predictions = []
    for i in range(steps):
        input_data = data[np.newaxis, :, np.newaxis]
        generated = model.predict(input_data)[0]
        data = np.append(data, generated)[-look_back:]
        predictions.append(generated)
    return predictions


class PlotCallback(keras.callbacks.Callback):
    def __init__(self, trainX, trainY, testX, testY, look_back, repeated_predictions=True):
        self.repeat_predictions = repeated_predictions
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.look_back = look_back

    def on_epoch_end(self, epoch, logs):
        if epoch % 10 != 0:
            return

        if self.repeat_predictions:
            preds = repeated_predictions(
                self.model, self.trainX[-1, :, 0], self.look_back, self.testX.shape[0])
        else:
            preds = self.model.predict(self.testX)

        # Generate a figure with matplotlib</font>
        figure = matplotlib.pyplot.figure(figsize=(5, 5))
        plot = figure.add_subplot(111)

        plot.plot(self.trainY)
        plot.plot(np.append(np.empty_like(self.trainY) * np.nan, self.testY))
        plot.plot(np.append(np.empty_like(self.trainY) * np.nan, preds))

        data = fig2data(figure)
        matplotlib.pyplot.close(figure)

        wandb.log({"image": wandb.Image(data)}, commit=False)
