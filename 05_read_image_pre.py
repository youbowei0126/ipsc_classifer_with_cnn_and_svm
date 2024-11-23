import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tifffile as tiff
import mylib as my
import pandas as pd
from tqdm import tqdm
from pathlib import Path

PATH=r"assest_image\908030_2.tiff"
tiff_data = tiff.imread(PATH)

my.plot_color_image(tiff_data)
plt.show()
