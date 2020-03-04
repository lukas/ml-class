import ipywidgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np
import wandb


class Model(object):
    """Base class for the other *Model classes.
    Implements plotting and interactive components
    and interface with Parameters object."""

    def __init__(self, input_values, model_inputs, parameters, funk):
        self.model_inputs = np.atleast_2d(model_inputs)
        self.input_values = input_values
        self.parameters = parameters
        self.funk = funk
        self.plotted = False
        self.has_data = False
        self.show_MSE = False

    def plot(self):
        if not self.plotted:
            self.initialize_plot()
        else:
            self.artist.set_data(self.input_values, self.outputs)
        return

    @property
    def outputs(self):
        return np.squeeze(self.funk(self.model_inputs))

    def initialize_plot(self):
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        self.artist, = self.ax.plot(
            self.input_values, self.outputs, linewidth=4,
            color=np.divide([255, 204, 51], 256))
        self.plotted = True
        self.ax.set_ylim([0, 1])
        self.ax.set_xlim([0, 1])

    def make_interactive(self, log=True):
        """called in a cell after Model.plot()
        to make the plot interactive."""

        self.log = log

        if self.log:
            wandb.init()

        @interact(**self.parameters.widgets)
        def make(**kwargs):

            for parameter in kwargs.keys():
                self.parameters.dict[parameter] = kwargs[parameter]
            self.parameters.update()
            self.plot()

            if self.log:
                wandb.log(kwargs)

            if self.show_MSE:
                MSE = self.compute_MSE()
                print("loss:\t"+str(MSE))
            if self.log:
                wandb.log({"loss": MSE})
            return

        return

    def set_data(self, xs, ys):
        self.data_inputs = self.transform_inputs(xs)
        self.correct_outputs = ys

        if self.has_data:
            _offsets = np.asarray([xs, ys]).T
            self.data_scatter.set_offsets(_offsets)
        else:
            self.data_scatter = self.ax.scatter(xs, ys,
                                                color='k', alpha=0.5, s=72)
            self.has_data = True

    def compute_MSE(self):
        """Used in fitting models lab to display MSE performance
        for hand-fitting exercises"""
        outputs = np.squeeze(self.funk(self.data_inputs))
        squared_errors = np.square(self.correct_outputs - outputs)
        MSE = np.mean(squared_errors)
        return MSE


class LinearModel(Model):
    """A linear model is a model whose transform is
    the dot product of its parameters with its inputs.
    Technically really an affine model, as LinearModel.transform_inputs
    adds a bias term."""

    def __init__(self, input_values, parameters, model_inputs=None):

        if model_inputs is None:
            model_inputs = self.transform_inputs(input_values)
        else:
            model_inputs = model_inputs

        def funk(inputs):
            return np.dot(self.parameters.values, inputs)

        Model.__init__(self, input_values, model_inputs, parameters, funk)

    def transform_inputs(self, input_values):
        model_inputs = [[1]*input_values.shape[0], input_values]
        return model_inputs


class Parameters(object):
    """Tracks and updates parameter values and metadata, like range and identity,
    for parameters of a model. Interfaces with widget-making tools
    via the Model class to make interactive widgets for Model plots."""

    def __init__(self, defaults, ranges, names=None):
        assert len(defaults) == len(ranges),\
            "must have default and range for each parameter"

        self.values = np.atleast_2d(defaults)

        self.num = len(defaults)

        self._zip = zip(defaults, ranges)

        if names is None:
            self.names = ['parameter_' + str(idx) for idx in range(self.num)]
        else:
            self.names = names

        self.dict = dict(zip(self.names, self.values))

        self.defaults = defaults
        self.ranges = ranges

        self.make_widgets()

    def make_widgets(self):
        self._widgets = [self.make_widget(parameter, idx)
                         for idx, parameter
                         in enumerate(self._zip)]

        self.widgets = {self.names[idx]: _widget
                        for idx, _widget
                        in enumerate(self._widgets)}

    def make_widget(self, parameter, idx):
        default = parameter[0]
        range = parameter[1]
        name = self.names[idx]
        return ipywidgets.FloatSlider(
            value=default, min=range[0], max=range[1],
            step=0.01, description=name)

    def update(self):
        sorted_keys = sorted(self.dict.keys())
        self.values = np.atleast_2d([self.dict[key] for key in sorted_keys])
