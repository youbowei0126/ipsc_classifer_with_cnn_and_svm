import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tifffile as tiff
import mylib as my
import pandas as pd
from tqdm import tqdm
from pathlib import Path


disk_ = "C:/"

raw_path = disk_ + "projects/allen_cell_data/raw_image"
seg_path = disk_ + "projects/allen_cell_data/segment_image"
output_path = disk_ + "projects/allen_cell_data/output_image"
Path(output_path).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(r"assest_csv\select_TUBA1B_sample.csv")

scaler_Struct = StandardScaler()
scaler_405 = StandardScaler()
scaler_638 = StandardScaler()


for i in [0]:
    i = 0
    row = df.iloc[i]
    img_path = raw_path + "\\" + str(row["CellId"]) + ".tiff"
    output_path = output_path + "\\" + str(row["CellId"]) + ".tiff"
    img = tiff.imread(img_path)
    print(img.shape)
    t = np.array(
        row[["ChannelNumberStruct", "ChannelNumber405", "ChannelNumber638"]]
    ).astype(np.int32)
    img_output = img[
        img.shape[0] // 3,
        t,
        :,
        :,
    ]
    if i == 0:
        scaler_Struct.fit(img_output[:, :, 0].reshape(-1, 1))
        scaler_405.fit(img_output[:, :, 1].reshape(-1, 1))
        scaler_638.fit(img_output[:, :, 2].reshape(-1, 1))
        
    img_output[:, :, 0] = scaler_Struct.transform(
        img_output[:, :, 0].reshape(-1, 1)
    ).reshape(img_output.shape[:2])
    
    img_output[:, :, 1] = scaler_405.transform(
        img_output[:, :, 1].reshape(-1, 1)
    ).reshape(img_output.shape[:2])
    
    img_output[:, :, 2] = scaler_638.transform(
        img_output[:, :, 2].reshape(-1, 1)
    ).reshape(img_output.shape[:2])

    img_output = np.transpose(img_output, (1, 2, 0))
    print(img_output.shape)
    tiff.imwrite(output_path, img_output.astype(np.float32), dtype=np.float32)
    


# my.plot_color_image(img_output)
# plt.show()
