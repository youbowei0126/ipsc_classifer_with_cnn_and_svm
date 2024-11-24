import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tifffile as tiff
import mylib as my
import pandas as pd
from tqdm import tqdm

def laplacian_variance(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian.var()


def plot_gray_image(image, new_frame=True):
    if new_frame:
        my.plt_general_setting_init()
    plt.imshow(image, cmap="binary", interpolation="nearest")
    # plt.axis("off")


PATH = r"dataset_new\iPSC_Morphologies\Round\Round_o0011_i6542_APC-Brightfield-DAPI-GREEN-PE-CellSegmentation-NucleusSegmentation.tiff"
# 讀取 TIFF 影像
tiff_data = tiff.imread(PATH)

# 顯示影像的形狀
print(tiff_data.shape)
# with tiff.TiffFile(PATH) as tif:
#     for page in tif.pages:
#         print(page.tags)  # 查看每一頁的標籤數據
my.plt_general_setting_init()
for i in range(tiff_data.shape[2]):
    plt.subplot(2, 4, i + 1)
    plot_gray_image(tiff_data[:,:,i], new_frame=False)
    # plt.title(tiff_data[:, i, :, :].max())
    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()







plt.show()
