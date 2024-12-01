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
    '''
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)]
            )
        except RuntimeError as e:
            print(e)'''
    

def bulid_model():
    
    def print_shape(x):
        print(x.shape)
        return x

  
    
    input_layer=layers.Input(shape=input_shape)
    new_h = input_shape[0]
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    new_w = input_shape[1]
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    # padded_input = layers.Lambda(print_shape)(input_layer)
    padded_input = layers.Lambda(lambda x: tf.image.resize_with_pad(x, new_h, new_w))(input_layer)
    # padded_input = layers.Lambda(print_shape)(padded_input)
    new_h_half=new_h//2
    new_w_half=new_w//2
    full=layers.Conv2D(cnn_feature[0], (3,3), activation='relu', padding='same')(padded_input)
    full=layers.MaxPooling2D((2,2), padding='same')(full)
    left_top = layers.Lambda(lambda x: x[:, :new_h_half, :new_w_half, :])(padded_input)
    right_top = layers.Lambda(lambda x: x[:, :new_h_half, new_w_half:, :])(padded_input)
    left_bottom = layers.Lambda(lambda x: x[:, new_h_half:, :new_w_half, :])(padded_input)
    right_bottom = layers.Lambda(lambda x: x[:, new_h_half:, new_w_half:, :])(padded_input)
    # right_bottom = layers.Lambda(print_shape)(right_bottom)
    # print(right_bottom.shape)
    def cnn_and_pool(input_layer, filters=cnn_feature, kernel_size=(3, 3), pool_size=(2, 2)):
        x = layers.Conv2D(filters[0], kernel_size, activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D(pool_size, padding='same')(x)
        x = layers.Conv2D(filters[1], kernel_size, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size, padding='same')(x)
        x = layers.Conv2D(filters[2], kernel_size, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size, padding='same')(x)
        return x
    
    full_cnn=cnn_and_pool(full)
    left_top_cnn=cnn_and_pool(left_top)
    right_top_cnn=cnn_and_pool(right_top)
    left_bottom_cnn=cnn_and_pool(left_bottom)
    right_bottom_cnn=cnn_and_pool(right_bottom)
    conjugate=layers.Concatenate(axis=-1)([full_cnn,left_top_cnn,right_top_cnn,left_bottom_cnn,right_bottom_cnn])
    output=layers.Conv2D(cnn_feature[3], (3,3), activation='relu', padding='same')(conjugate)
    output=layers.MaxPooling2D((2,2), padding='same')(output)
    
    output=layers.Flatten()(output)
    
    for i in hidden_layer:
        output=layers.Dense(units=i, activation="relu")(output)
        output=layers.Dropout(0.2)(output)
    output=layers.Dense(units=n_catagory, activation="softmax")(output)
    
    return Model(inputs=input_layer, outputs=output)


def main():
    print("program start")
    set_random_seed(42)
    check_GPU()

    

    global hidden_layer;hidden_layer = [1024, 1024]
    global cnn_feature;cnn_feature = [32, 64, 128, 256]
    global input_shape;input_shape = (150, 150, 7)
    global n_catagory;n_catagory = 5
    global train_new_model;train_new_model=True
    global print_model_blueprint;print_model_blueprint=True
    
    # train_new_model=False to use trained model
    
    
    #  build model
    model=bulid_model()
    from tensorflow.keras.optimizers import RMSprop
    # RMSprop(learning_rate=1e-4)
    model.compile(optimizer=RMSprop(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    # plot model
    from tensorflow.keras.utils import plot_model
    if print_model_blueprint:
        print("generate image...")
        plot_model(model,show_shapes=True,to_file="model004.png",dpi=2400)
        print("generate image complete")
    
    
    
    
    # evaluate model
    
    x_train, y_train, x_test, y_test, list_of_cata = load_dataset(r"dataset_new\iPSC_Morphologies")
    print(list_of_cata)
    print(x_train.shape)
    print(y_train.shape)

    '''
    from sklearn.preprocessing import StandardScaler
    # stat_df=pd.read_csv(r"calculate_dataset_Statistics.csv")
    for i in tqdm(range(x_train.shape[-1])):
        scaler=StandardScaler()
        # scaler.mean_=stat_df.loc[i,"average"]
        # scaler.scale_=stat_df.loc[i,"std"]
        temp=scaler.fit_transform(x_train[:,:,:,i].reshape(-1, 1))
        temp=temp.reshape(x_train[:,:,:,i].shape)
        x_train[:,:,:,i]=temp
        
        temp=scaler.transform(x_test[:,:,:,i].reshape(-1, 1))
        temp=temp.reshape(x_test[:,:,:,i].shape)
        x_test[:,:,:,i]=temp
    print(x_train.shape)'''
    
    
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
        model.save("model004.keras")
        plot_train_history(train_history)
    else:
        model=load_model("model004.keras")
        print("load trained model")
     
    from sklearn.metrics import classification_report
    loss, accuracy = model.evaluate(x_test, y_test)
    
    print("test loss: ", loss)
    print("test accuracy: ", accuracy)
    from sklearn.metrics import classification_report
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1)))
    # print(model.layers)
    
    
    # get cnn model
    for i in model.layers:
        if type(i)==type(layers.Flatten()):
            CNN_model=Model(inputs=model.input,outputs=i.output)
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
