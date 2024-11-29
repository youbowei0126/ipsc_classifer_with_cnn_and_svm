import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tifffile as tiff
import mylib as my
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tifffile import imread
from keras import Sequential, layers
import os


def plot_train_history(train_history, train="accuracy", validation="val_accuracy"):
    my.plt_general_setting_init()
    plt.plot(train_history.history[train], label="train")
    plt.plot(train_history.history[validation], label="validation")
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    my.plt_general_setting_end("train_history.svg")


def readDataset(path):
    train_path = path + "/" + "train/"
    test_path = path + "/" + "test/"
    cata = os.listdir(train_path)
    cata_number = len(cata)
    # if cata == os.listdir(test_path):
    #     print("test and train is not compatible")
    # x_train,y_train,x_test,y_test=np.empty((0,0,0,0)),np.empty((0,0)),np.empty((0,0,0,0)),np.empty((0,0))
    list_of_cata = []
    for i in range(len(cata)):
        list_of_cata.append(cata)
        if i == 0:
            x_train_this_cata = readFileFromFolder(train_path + "/" + cata[i])
            x_train = x_train_this_cata
            y_train_this_cata = np.zeros((len(x_train_this_cata), cata_number))
            y_train_this_cata[:, i] = 1
            y_train = y_train_this_cata

            x_test_this_cata = readFileFromFolder(test_path + "/" + cata[i])
            x_test = x_test_this_cata
            y_test_this_cata = np.zeros((len(x_test_this_cata), cata_number))
            y_test_this_cata[:, i] = 1
            y_test = y_test_this_cata
            print("load {c} complete")
        else:
            x_train_this_cata = readFileFromFolder(train_path + "/" + cata[i])
            x_train = np.vstack((x_train, x_train_this_cata))
            y_train_this_cata = np.zeros((len(x_train_this_cata), cata_number))
            y_train_this_cata[:, i] = 1
            y_train = np.vstack((y_train, y_train_this_cata))

            x_test_this_cata = readFileFromFolder(test_path + "/" + cata[i])
            x_test = np.vstack((x_test, x_test_this_cata))
            y_test_this_cata = np.zeros((len(x_test_this_cata), cata_number))
            y_test_this_cata[:, i] = 1
            y_test = np.vstack((y_test, y_test_this_cata))
            print("load {c} complete")

    return x_train, y_train, x_test, y_test, list_of_cata


def readFileFromFolder(folder):
    # List all file and read to array
    files = os.listdir(folder)
    all_img = []
    for tif in tqdm(files):
        if tif[-5:] == ".tiff":
            img = imread(folder + "/" + tif)
            all_img.append(img)
        # print(folder + "/" + tif)
    all_img = np.array(all_img)
    return all_img


def set_random_seed(i):
    # random.seed(42)
    np.random.seed(i)
    tf.random.set_seed(i)
    os.environ["PYTHONHASHSEED"] = f"{i}"


def check_GPU():
    from tensorflow.python.client import device_lib

    device_lib.list_local_devices()
    print(tf.config.experimental.list_physical_devices("GPU"))
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")


def main():
    print("program start")
    set_random_seed(42)
    # import tensorflow as tf
    check_GPU()
    x_train, y_train, x_test, y_test, list_of_cata = readDataset(
        r"dataset_new\iPSC_QCData"
    )

    # x_batch, y_batch = next(train_generator)
    # train_generator.reset()
    # print("x_batch.shape=",x_batch.shape)
    # print("y_batch.shape=",y_batch.shape)

    hidden_layer = [1024, 1024]
    cnn_feature = [32, 64, 128]
    model = Sequential()
    model.add(
        layers.Convolution2D(
            cnn_feature[0],
            (3, 3),
            input_shape=(100, 100, 5),
            activation="relu",
            padding="SAME",
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(
        layers.Convolution2D(cnn_feature[1], (3, 3), activation="relu", padding="SAME")
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(
        layers.Convolution2D(cnn_feature[2], (3, 3), activation="relu", padding="SAME")
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hidden_layer[0], activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=hidden_layer[1], activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=4, activation="softmax"))
    from tensorflow.keras.optimizers import RMSprop

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    train_history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=30,
        verbose=1,
        validation_split=0.2,
    )
    model.save("model001.keras")
    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: ", loss)
    print("test accuracy: ", accuracy)

    plot_train_history(train_history)

    print("program end")


# x_train,y_train,x_test,y_test=readDataset(r"dataset_new\iPSC_QCData")
# print(x_train.shape)
main()
