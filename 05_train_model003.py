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


def plot_train_history(train_history, train="accuracy", validation="val_accuracy"):
    my.plt_general_setting_init()
    plt.plot(train_history.history[train], label="train")
    plt.plot(train_history.history[validation], label="validation")
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    my.plt_general_setting_end("train_history.svg")


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


def set_random_seed(i):
    """
    https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
    """

    # random.seed(42)
    np.random.seed(i)
    tf.random.set_seed(i)
    tf.compat.v2.random.set_seed(i)
    os.environ["PYTHONHASHSEED"] = str(i)
    # set_global_determinism
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    print("seed set")


def check_GPU():
    from tensorflow.python.client import device_lib

    device_lib.list_local_devices()
    print(tf.config.experimental.list_physical_devices("GPU"))
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")


def bulid_model(model):
    model.add(layers.Convolution2D(cnn_feature[0],(3, 3),input_shape=input_shape,activation="relu",padding="SAME",))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Convolution2D(cnn_feature[1], (3, 3), activation="relu", padding="SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Convolution2D(cnn_feature[2], (3, 3), activation="relu", padding="SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hidden_layer[0], activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=hidden_layer[1], activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=n_catagory, activation="softmax"))
    return model


def main():
    print("program start")
    set_random_seed(1)
    check_GPU()

    #  build model

    global hidden_layer;hidden_layer = [1024, 1024]
    global cnn_feature;cnn_feature = [32, 64, 128]
    global input_shape;input_shape = (150, 150, 7)
    global n_catagory;n_catagory = 5
    global train_new_model;train_new_model=False
    # train_new_model=False to use trained model
    
    
    
    model = Sequential()
    model=bulid_model(model)
    from tensorflow.keras.optimizers import RMSprop
    
    model.compile(optimizer=RMSprop(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    # plot model
    from tensorflow.keras.utils import plot_model,model_to_dot
    plot_model(model,show_shapes=True,to_file="model002.png",dpi=2400)
        
    # evaluate model

    x_train, y_train, x_test, y_test, list_of_cata = load_dataset(r"dataset_new\iPSC_Morphologies")
    print(list_of_cata)
    

    if train_new_model:
        print("start training")
        train_history = model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=30,
            verbose=1,
            validation_split=0.2,
        )
        model.save("model002.keras")
        plot_train_history(train_history)
    else:
        model=load_model("model002.keras")
        print("load trained model")
     
    from sklearn.metrics import classification_report
    loss, accuracy = model.evaluate(x_test, y_test)
    
    print("test loss: ", loss)
    print("test accuracy: ", accuracy)

    
    # get cnn model
    CNN_model=Model(inputs=model.input,outputs=model.layers[6].output)
    CNN_model.summary()
    features_train = CNN_model.predict(x_train)
    features_test = CNN_model.predict(x_test)
    y_train_cata=np.argmax(y_train, axis=1)
    y_test_cata=np.argmax(y_test, axis=1)
    
    print(features_train.shape)
    print(features_train[0])
    
    # train svm
    svm = SVC(kernel='linear')
    print("start training svm...")
    svm.fit(features_train, y_train_cata)
    print("training end")
    y_predict=svm.predict(features_test)
    
    report = classification_report(y_test_cata, y_predict)
    print(report)
    plt.show()
    
    

    
    
    print("program end")


# x_train,y_train,x_test,y_test=readDataset(r"dataset_new\iPSC_QCData")
# print(x_train.shape)
main()
