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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")


class PlotCallback(keras.callbacks.Callback):
    def __init__(self, encoder, decoder, data):
        self.encoder = encoder
        self.decoder = decoder
        self.x_test, self.y_test = data
        self.batch_size = 64

    def on_epoch_end(self, epoch, logs):

        # Generate a figure with matplotlib</font>
        figure = matplotlib.pyplot.figure(figsize=(10, 10))
        plt = figure.add_subplot(111)

        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(self.x_test,
                                            batch_size=self.batch_size)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=self.y_test)
        # plt.colorbar()
        # plt.xlabel("z[0]")
        # plt.ylabel("z[1]")

        data = fig2data(figure)
        matplotlib.pyplot.close(figure)

        wandb.log({"scatter": wandb.Image(data)}, commit=False)

        # Generate a figure with matplotlib</font>
        figure = matplotlib.pyplot.figure(figsize=(10, 10))
        plt = figure.add_subplot(111)

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        fig = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                fig[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        #plt.xticks(pixel_range, sample_range_x)
        #plt.yticks(pixel_range, sample_range_y)
        # plt.xlabel("z[0]")
        # plt.ylabel("z[1]")
        plt.imshow(fig, cmap='Greys_r')

        data = fig2data(figure)
        matplotlib.pyplot.close(figure)

        wandb.log({"grid": wandb.Image(data)}, commit=False)
