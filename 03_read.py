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


PATH = r"assest_image\234004.tiff"
# 讀取 TIFF 影像
tiff_data = tiff.imread(PATH)

# 顯示影像的形狀
print(tiff_data.shape)
# with tiff.TiffFile(PATH) as tif:
#     for page in tif.pages:
#         print(page.tags)  # 查看每一頁的標籤數據
my.plt_general_setting_init()
for i in range(tiff_data.shape[1]):
    plt.subplot(2, 4, i + 1)
    plot_gray_image(tiff_data[tiff_data.shape[0] // 3, i, :, :], new_frame=False)
    plt.axis("off")

plt.tight_layout()


for j in tqdm(range(tiff_data.shape[1])):
    my.plt_general_setting_init()
    for i in range(tiff_data.shape[0]):
        plt.subplot(9,10,i+1)
        plot_gray_image(tiff_data[i,j,:,:],new_frame=False)
        plt.axis("off")
        # plt.title(f"{(tiff_data[i,j,:,:].max()-tiff_data[i,j,:,:].min())}",fontsize=7)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

with open("tag.txt", "w") as file_:
    with tiff.TiffFile(PATH) as tif:
        for page in tif.pages:
            print(page.tags, file=file_)  # 查看每一頁的標籤數據

info = pd.read_csv(r"assest_csv\select_TUBA1B_sample.csv")
print(
    info.loc[
        3,
        [
            "ChannelNumberStruct",
            "ChannelNumberBrightfield",
            "ChannelNumber405",
            "ChannelNumber638",
        ],
    ]
)


plt.show()
