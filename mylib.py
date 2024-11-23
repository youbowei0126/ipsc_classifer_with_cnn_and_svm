import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import tensorflow as tf


def parent_folder_or_file_under(input_=""):
    # import os
    if input_ == "":
        return os.path.dirname(__file__)
    else:
        return os.path.join(os.path.dirname(__file__), input_)


def plt_general_setting_init(figsize=(12, 5), fontsize=12):
    plt.rcParams.update({"font.size": fontsize})
    plt.rc("font", family="Consolas")
    plt.figure(figsize=figsize)


def plt_general_setting_end(path_=None, axis_eq=False):
    plt.legend()
    if axis_eq:
        plt.axis("equal")
    plt.tight_layout()
    plt.grid()
    if path_ is None:
        path_ = parent_folder_or_file_under("figure0.svg")
    else:
        path_ = parent_folder_or_file_under(path_)
        plt.savefig(path_)

    # plt.show()


def legend_msg(msg):
    plt.plot([], [], " ", label=msg)


def plot_gray_image(image, new_frame=True):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if new_frame:
        plt_general_setting_init()
    plt.imshow(image, cmap="binary", interpolation="nearest")
    # plt.axis("off")


def plot_color_image(image, new_frame=True):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if new_frame:
        plt_general_setting_init()
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    # plt.axis("off")
