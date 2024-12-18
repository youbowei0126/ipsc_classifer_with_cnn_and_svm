import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import sklearn.preprocessing as standardscaler
import pandas as pd

# 讀取 tiff 圖像（5 個通道）


def show_image(input_path_, output_path_,ylabel_):
    plt.figure(figsize=(10,2))
    img = tiff.imread(input_path_)  # shape: (100, 100, 5)
    img = img.astype(np.float32)
    w = [[2, 0, 0], [2, 2, 2], [0, 0, 8], [0, 3, 0], [100,100, 0]]
    ch_max = pd.read_csv(r"calculate_dataset_Statistics.csv")
    ch_max = ch_max["max"]
    ch_max = ch_max.to_numpy()
    print(ch_max)
    ch_max = ch_max
    
    plt.subplot(1, 6, 1)  # Increase the number of columns to 6 to accommodate the ylabel
    plt.axis("off")  # Turn off axis for the label subplot
    plt.text(0.5, 0.5, ylabel_, fontsize=18, ha='center', va='center')  # Position the label in the center of the plot

    
    
    for i in range(5):
        print(img[:, :, i])
        img[:, :, i] = img[:, :, i] / ch_max[i]
        print(img[:, :, i])
        red = img[:, :, i] * w[i][0]
        green = img[:, :, i] * w[i][1]
        blue = img[:, :, i] * w[i][2]
        rgb_image = np.dstack((red, green, blue))
        plt.subplot(1, 6, i + 2)
        plt.imshow(rgb_image)
        plt.axis("off")
    plt.savefig(output_path_, dpi=600, transparent=True)
    


show_image(
    r"F:\final_project\dataset_new\iPSC_Morphologies\train\Big\Big_o0009_i4008_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"visualize\dataset2\Big.png",
    "Big"
)
show_image(
    r"dataset_new\iPSC_Morphologies\train\Long\Long_o0250_i3526_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"visualize\dataset2\Long.png",
    "Long"
)
show_image(
    r"dataset_new\iPSC_Morphologies\train\Mitotic\Mitotic_o0016_i4046_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"visualize\dataset2\Mitotic.png",
    "Mitotic"
)
show_image(
    r"dataset_new\iPSC_Morphologies\train\RAR-treated\RAR-treated_o0076_i0539_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"visualize\dataset2\RAR-treated.png",
    "RAR-treated"
)
show_image(
    r"dataset_new\iPSC_Morphologies\train\Round\Round_o0005_i5039_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff",
    r"visualize\dataset2\Round.png",
    "Round"
)
plt.show()

