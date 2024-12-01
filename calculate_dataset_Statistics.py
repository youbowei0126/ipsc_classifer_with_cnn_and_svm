import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mylib as my
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras import Sequential, layers
import os
import sys
from tensorflow.keras.models import Model,load_model
from sklearn.svm import SVC
def load_dataset(path):
    print('start loading dataset ...')
    train_path = path + "/" + "train/"
    test_path = path + "/" + "test/"
    cata = os.listdir(train_path)
    cata_number = len(cata)
    # if cata == os.listdir(test_path):
    #     print("test and train is not compatible")
    # x_train,y_train,x_test,y_test=np.empty((0,0,0,0)),np.empty((0,0)),np.empty((0,0,0,0)),np.empty((0,0))
    list_of_cata = []
    for i in range(len(cata)):
        list_of_cata.append(cata[i])
        if i == 0:
            x_train_this_cata = read_TFF_file_from_folder(train_path + "/" + cata[i])
            x_train = x_train_this_cata
            y_train_this_cata = np.zeros((len(x_train_this_cata), cata_number))
            y_train_this_cata[:, i] = 1
            y_train = y_train_this_cata

            x_test_this_cata = read_TFF_file_from_folder(test_path + "/" + cata[i])
            x_test = x_test_this_cata
            y_test_this_cata = np.zeros((len(x_test_this_cata), cata_number))
            y_test_this_cata[:, i] = 1
            y_test = y_test_this_cata
            print(f"load \"{cata[i]}\" complete")
        else:
            x_train_this_cata = read_TFF_file_from_folder(train_path + "/" + cata[i])
            x_train = np.vstack((x_train, x_train_this_cata))
            y_train_this_cata = np.zeros((len(x_train_this_cata), cata_number))
            y_train_this_cata[:, i] = 1
            y_train = np.vstack((y_train, y_train_this_cata))

            x_test_this_cata = read_TFF_file_from_folder(test_path + "/" + cata[i])
            x_test = np.vstack((x_test, x_test_this_cata))
            y_test_this_cata = np.zeros((len(x_test_this_cata), cata_number))
            y_test_this_cata[:, i] = 1
            y_test = np.vstack((y_test, y_test_this_cata))
            print(f"load \"{cata[i]}\" complete")

    return x_train, y_train, x_test, y_test, list_of_cata


def read_TFF_file_from_folder(folder):
    # List all file and read to array
    from tifffile import imread

    files = os.listdir(folder)
    all_img = []
    for tif in tqdm(files):
        if tif[-5:] == ".tiff":
            img = imread(folder + "/" + tif)
            all_img.append(img)
        # print(folder + "/" + tif)
    all_img = np.array(all_img)
    return all_img

def main():
    x_train, y_train, x_test, y_test, list_of_cata = load_dataset(r"dataset_new\iPSC_Morphologies")
    print(list_of_cata)
    n_channel=x_train.shape[-1]
    stat_df=pd.DataFrame([])
    stat_df["channel"]=np.array(range(n_channel))
    for i in tqdm(range(n_channel)):
        data=x_train[:,:,:,i].flatten()
        ave=data.mean()
        std=data.std()
        max=data.max()
        min=data.min()
        stat_df.loc[i,"average"]=ave
        stat_df.loc[i,"std"]=std
        stat_df.loc[i,"max"]=max
        stat_df.loc[i,"min"]=min
        
    stat_df.to_csv("calculate_dataset_Statistics.csv")
        
        
    
    
main()