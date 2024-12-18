import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tifffile as tiff
from tensorflow.keras.models import Model, load_model


def heatmap(a_img, model, layer_name, channel, limit_id, show_color_bar=False,show_color_bar_scale=True):
    # Create a model that outputs the activations of the chosen layer
    layer_output = model.get_layer(layer_name).output
    feature_model = Model(inputs=model.input, outputs=layer_output)

    # load image

    # print(a_img.shape)

    # Get activations from the chosen layer
    activations = feature_model.predict(np.array([a_img]))
    print(layer_name,activations.shape)
    # Choose which filter/channel to visualize (e.g., first filter of the first convolutional layer)
    activation_map = activations[0, :, :, channel]

    # Plot the activation map (heatmap)
    plt.imshow(activation_map, cmap="jet", vmin=limit[limit_id][0], vmax=limit[limit_id][1])
    plt.axis("off")
    if show_color_bar:
        colorbar = plt.colorbar()  # Show color scale
        if not(show_color_bar_scale):
            colorbar.ax.axis("off")

    # plt.title(f"Feature Extraction Heatmap from {layer_name}")


def heatmap4(img_path, model, layer_name, img_channel,filter_channel, row, total_row):
    a_img = tiff.imread(img_path)
    plt.subplot(total_row, len(layer_name) + 1, (row - 1) * (len(layer_name) + 1) + 1)
    plt.imshow(
        np.dstack(
            (
                a_img[:, :, img_channel] * 0,
                a_img[:, :, img_channel] * 3 / 2375,
                a_img[:, :, img_channel] * 0,
            )
        ),
    )
    plt.axis("off")

    for i in range(len(layer_name)):
        plt.subplot(
            total_row, len(layer_name) + 1, (row - 1) * (len(layer_name) + 1) + i + 2
        )
        heatmap(
            a_img,
            model,
            layer_name[i],
            channel=filter_channel,
            limit_id=i + 1,
            show_color_bar=True,
        )


# Load pre-trained model (if you already have a trained model, replace VGG16 with your model)
model = load_model(r"model005.keras")

# Choose the layer to extract features from (e.g., first convolutional layer)
layer_name = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]  # Example for VGG16
img_channel = 3
filter_channel=3
img_path_ = r"F:\final_project\dataset_new\iPSC_Morphologies\train\Big\Big_o0009_i4008_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff"
limit = [[0, 64876], [0, 5000], [0, 3000], [0, 2000], [0, 1000]]
img_path_ = [
    r"F:\final_project\dataset_new\iPSC_Morphologies\train\Big\Big_o0009_i4008_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"dataset_new\iPSC_Morphologies\train\Long\Long_o0250_i3526_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"dataset_new\iPSC_Morphologies\train\Mitotic\Mitotic_o0016_i4046_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"dataset_new\iPSC_Morphologies\train\RAR-treated\RAR-treated_o0076_i0539_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"dataset_new\iPSC_Morphologies\train\Round\Round_o0005_i5039_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
]

plt.figure(figsize=(15, 10))
totalrow = len(img_path_)
for i in range(totalrow):
    heatmap4(img_path_[i], model, layer_name, img_channel,filter_channel, i + 1, totalrow)
plt.savefig(r"visualize/heatmap/heatmap_ch3.png", transparent=True, dpi=600)
plt.show()
