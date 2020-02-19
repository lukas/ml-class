import ipywidgets
from ipywidgets import interact
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy.signal import correlate2d

WANDB_GOLD = np.divide([255, 205, 50], 255)


def display_convolution(image, kernel, relu=False, axs=None):
    if axs is None:
        fig, axs = plt.subplots(ncols=3, figsize=(9, 4.5))
    image = (image - np.mean(image)) / np.std(image)
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Image")
    
    kernel = np.divide(kernel, np.linalg.norm(kernel, ord="fro"))
    
    axs[1].imshow(kernel, cmap='gray')
    axs[1].set_title("Filter/Kernel/Conv2D\nWeights")
    
    raw_features = correlate2d(image, kernel, mode="valid")
    features = np.copy(raw_features)
    
    if relu:
        features = np.where(features > 0, features, 0)
    
    #Normalize between 0.0 and 1.0
    features = (features - np.min(features)) / np.ptp(features)
    
    axs[2].imshow(features, cmap='gray')
    axs[2].set_title("Feature Map")
    [ax.axis("off") for ax in axs]
    
    return axs, image, raw_features, features


def make_interactive_convplots(image, kernel):
    kernel = np.array(kernel)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    _, image, raw_features, features = display_convolution(image, kernel, relu=False, axs=axs[0, :])
    image_ax, feature_ax = axs[0, 0], axs[0, -1]
    
    kernel_h, kernel_w = kernel.shape
    image_h, image_w = image.shape
    
    kernel_on_image = add_kernel_to_image(kernel, image_ax)
    kernel_on_feature = add_feature_to_image(feature_ax)
    
    zoomed_image_ax = axs[1, 0]
    pointwise_mult_ax = axs[1, 1]
    zoomed_feature_ax = axs[1, 2]
    
    def compute_inputs(x, y):
        return np.multiply(image[y:y+kernel_h, x:x+kernel_w], kernel)
    
    zoomed_im = zoomed_image_ax.imshow(image[:kernel_h, :kernel_w],
                                       vmin=np.min(image), vmax=np.max(image),
                                       cmap='gray')
    pointwise_mult = pointwise_mult_ax.imshow(compute_inputs(0, 0),
                                              vmin=np.min(raw_features), vmax=np.max(raw_features), cmap='gray')
    feature_val = features[0, 0]
    zoomed_feature = zoomed_feature_ax.imshow(np.atleast_2d(feature_val),
                                              vmin=np.min(features), vmax=np.max(features),
                                              cmap='gray')
    feature_val_string = "{:03.1f}"
    feature_value_text = zoomed_feature_ax.text(
        0.5, 0.5, feature_val_string.format(feature_val),
        horizontalalignment='center', verticalalignment='center', transform=zoomed_feature_ax.transAxes,
        fontsize="x-large", fontweight="bold", color=WANDB_GOLD)
    
    [ax.axis('off') for ax in axs[1, :]]
    
    zoomed_image_ax.set_title("Current Input to Convolution")
    pointwise_mult_ax.set_title("Kernel x Current Input")
    zoomed_feature_ax.set_title("Current Feature Map Value")

    ymax = image_h - kernel_h
    x_slider = ipywidgets.IntSlider(0, 0, image_w - kernel_w)
    y_slider = ipywidgets.IntSlider(ymax, 0, ymax, orientation='vertical')
    
    @interact
    def update_convplots(x=x_slider, y=y_slider):
        y = ymax - y
        kernel_on_image.set_xy((x - 0.5, y - 0.5))
        kernel_on_feature.set_xy((x - 0.5, y - 0.5))
        
        zoomed_im.set_data(image[y:y+kernel_h, x:x+kernel_w])
        pointwise_mult.set_data(compute_inputs(x, y))
        feature_val = features[y, x]
        zoomed_feature.set_data(np.atleast_2d(feature_val))
        feature_value_text.set_text(feature_val_string.format(feature_val))
        fig.canvas.draw()
        
    return fig, axs, kernel_on_image


def add_kernel_to_image(kernel, image_ax):
    kernel_h, kernel_w = kernel.shape
    
    kernel_artist = matplotlib.patches.Rectangle(
        xy=(0, kernel_h), width=kernel_w, height=kernel_h,
        alpha=1, facecolor="none", edgecolor=WANDB_GOLD, lw=2)
    
    image_ax.add_artist(kernel_artist)
    
    return kernel_artist


def add_feature_to_image(feature_ax):
    
    feature_artist = matplotlib.patches.Rectangle(
        xy=(0, 1), width=1, height=1,
        alpha=1, facecolor="none", edgecolor=WANDB_GOLD, lw=1)
    
    feature_ax.add_artist(feature_artist)
    
    return feature_artist