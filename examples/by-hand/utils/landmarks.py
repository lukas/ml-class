from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import wandb


class LocalLandmarks(object):

    def __init__(self, points, num=200):
        self.points = np.array(points)
        self.grid = np.linspace(0, 1, num=num)
        self.pred_fun = np.array([self.grid, self.predict(self.grid)]).T

    def predict(self, xs):
        idxs_closest_x = []
        for x in xs:
            idxs_closest_x.append(np.argmin(np.abs(self.points[:, 0] - x)))
        return self.points[idxs_closest_x, 1]


class LandmarksModel(object):
    """Interactively build a landmark-based model.
    """
    def __init__(self, landmarks_polygon, xs, ys, train_size=15, log=True):
        self.landmarks_polygon = landmarks_polygon
        self.xp = list(landmarks_polygon.get_xdata())
        self.yp = list(landmarks_polygon.get_ydata())

        self.canvas = landmarks_polygon.figure.canvas
        self.ax_main = landmarks_polygon.axes

        self.observed_xs = xs[:train_size]
        self.observed_ys = ys[:train_size]

        self.make_scatter(self.observed_xs, self.observed_ys, self.ax_main)
        self.test_xs = xs[train_size:]
        self.test_ys = ys[train_size:]

        self.log = log

        self.cid = self.canvas.mpl_connect('button_press_event', self)

        prediction_line = Line2D([], [],
                                 c=np.divide([255, 204, 51], 256), lw=4)
        self.prediction_line_plot = self.ax_main.add_line(prediction_line)

        if self.log:
            wandb.init()

    def __call__(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.landmarks_polygon.axes:
            return

        # Add point
        self.xp.append(event.xdata)
        self.yp.append(event.ydata)

        self.landmarks_polygon.set_data(self.xp, self.yp)

        # Rebuild prediction curve and update canvas
        self.prediction_line_plot.set_data(*self._rebuild_predictor())
        self._update()

    def _update(self):
        self.canvas.draw()

    def _rebuild_predictor(self):
        self.local_landmarks = LocalLandmarks(list(zip(self.xp, self.yp)))

        train_MSE = self.compute_MSE(self.observed_xs, self.observed_ys)

        if self.log:
            test_MSE = self.compute_MSE(self.test_xs, self.test_ys)

            wandb.log({"train_loss": train_MSE,
                       "test_loss": test_MSE})

        x, y = self.local_landmarks.pred_fun.T

        return x, y

    def compute_MSE(self, xs, ys):
        predictions = self.local_landmarks.predict(xs)
        MSE = np.mean(np.square(ys - predictions))
        return MSE

    def make_scatter(self, xs, ys, ax):
        ax.scatter(xs, ys, color='k', alpha=0.5, s=72)


def setup_plot():
    fig, ax1 = plt.subplots(figsize=(10, 10))

    line = Line2D([], [], ls='none', c='#616666',
                  marker='x', mew=4, mec='k', ms=10, zorder=3)
    ax1.add_line(line)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    return line
