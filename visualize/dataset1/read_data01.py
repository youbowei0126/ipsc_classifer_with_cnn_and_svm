import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import sklearn.preprocessing as standardscaler
import pandas as pd

# 讀取 tiff 圖像（5 個通道）


def show_image(input_path_, output_path_):
    plt.figure(figsize=(10,2))
    img = tiff.imread(input_path_)  # shape: (100, 100, 5)
    img = img.astype(np.float32)
    w = [[2, 0, 0], [2, 2, 2], [0, 0, 5], [0, 5, 0], [5, 2.5, 0]]
    w = np.array(w, dtype=np.float32)
    ch_max = pd.read_csv(r"visualize\calculate_dataset1_Statistics.csv")[
        "max"
    ].to_numpy()
    print(ch_max)
    ch_max = ch_max
    for i in range(5):
        print(img[:, :, i])
        img[:, :, i] = img[:, :, i] / ch_max[i]
        print(img[:, :, i])
        red = img[:, :, i] * w[i][0]
        green = img[:, :, i] * w[i][1]
        blue = img[:, :, i] * w[i][2]
        rgb_image = np.dstack((red, green, blue))
        plt.subplot(1, 5, i + 1)
        plt.imshow(rgb_image)
        plt.axis("off")
    plt.savefig(output_path_, dpi=600, transparent=True)
    plt.show()


show_image(
    r"dataset_new\iPSC_QCData\train\Cell\Cell_o0003_i0738_APC-Brightfield-DAPI-GREEN-PE-RDP3001.tiff",
    r"visualize\dataset1\cell.png",
)
show_image(
    r"dataset_new\iPSC_QCData\train\Debris\Debris_o0001_i3456_APC-Brightfield-DAPI-GREEN-PE-RDP3001.tiff",
    r"visualize\dataset1\Debris.png",
)
show_image(
    r"dataset_new\iPSC_QCData\train\DyingCell\DyingCell_o0001_i2008_APC-Brightfield-DAPI-GREEN-PE-RDP3001.tiff",
    r"visualize\dataset1\DyingCell.png",
)
show_image(
    r"dataset_new\iPSC_QCData\train\MitoticCell\MitoticCell_o0001_i7476_APC-Brightfield-DAPI-GREEN-PE-RDP3001.tiff",
    r"visualize\dataset1\MitoticCell.png",
)
