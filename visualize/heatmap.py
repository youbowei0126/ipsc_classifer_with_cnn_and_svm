import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tifffile as tiff
from tensorflow.keras.models import Model, load_model


def heatmap(a_img,model,layer_name,channel,show_color_bar=False):
    # Create a model that outputs the activations of the chosen layer
    layer_output = model.get_layer(layer_name).output
    feature_model = Model(inputs=model.input, outputs=layer_output)

    # load image

    print(a_img.shape)

    # Get activations from the chosen layer
    activations = feature_model.predict(np.array([a_img]))

    # Choose which filter/channel to visualize (e.g., first filter of the first convolutional layer)
    activation_map = activations[0, :, :, channel]

    # Plot the activation map (heatmap)
    plt.imshow(activation_map, cmap="jet")
    plt.axis("off")
    if show_color_bar:
        colorbar=plt.colorbar()  # Show color scale
        colorbar.ax.axis("off")

    # plt.title(f"Feature Extraction Heatmap from {layer_name}")


# Load pre-trained model (if you already have a trained model, replace VGG16 with your model)
model = load_model(r"model005.keras")

# Choose the layer to extract features from (e.g., first convolutional layer)
layer_name = ["conv2d","conv2d_1","conv2d_2","conv2d_3"]  # Example for VGG16
channel=1
img_path_=r"F:\final_project\dataset_new\iPSC_Morphologies\train\Big\Big_o0009_i4008_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff"

def heatmap4(img_path,model,layer_name,channel):
    a_img = tiff.imread(img_path)
    plt.subplot(1,len(layer_name)+1,1)
    plt.imshow(a_img[:,:,0],cmap="binary")
    plt.axis("off")

    for i in range(len(layer_name)):
        plt.subplot(1,len(layer_name)+1,i+2)
        heatmap(a_img,model,layer_name[i],channel=channel,show_color_bar=False)


heatmap4(img_path_,model)
plt.show()






