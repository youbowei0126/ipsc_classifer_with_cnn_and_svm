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
from sklearn.preprocessing import StandardScaler
import joblib

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
        for i in range(1,len(cnn_feature)-1):
            x = layers.Conv2D(filters[i], kernel_size, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D(pool_size, padding='same')(x)
        return x
    
    full_cnn=cnn_and_pool(full)
    left_top_cnn=cnn_and_pool(left_top)
    right_top_cnn=cnn_and_pool(right_top)
    left_bottom_cnn=cnn_and_pool(left_bottom)
    right_bottom_cnn=cnn_and_pool(right_bottom)
    conjugate=layers.Concatenate(axis=-1)([full_cnn,left_top_cnn,right_top_cnn,left_bottom_cnn,right_bottom_cnn])
    output=layers.Conv2D(cnn_feature[len(cnn_feature)-1], (3,3), activation='relu', padding='same')(conjugate)
    output=layers.MaxPooling2D((2,2), padding='same')(output)
    
    output=layers.Flatten()(conjugate)
    
    for i in hidden_layer:
        output=layers.Dense(units=i, activation="relu")(output)
        output=layers.Dropout(0.2)(output)
    output=layers.Dense(units=n_catagory, activation="softmax")(output)
    
    return Model(inputs=input_layer, outputs=output)


def main(input_seed):
    print("program start")
    set_random_seed(input_seed)
    check_GPU()

    
    
    ######## setting
    global hidden_layer;hidden_layer = [1024,1024]
    global cnn_feature;cnn_feature = [32, 64, 128, 256]
    global input_shape;input_shape = (150, 150, 7)
    global n_catagory;n_catagory = 5
    global train_new_model;train_new_model=True
    global print_model_blueprint;print_model_blueprint=False
    global epochs;epochs=30
    global blueprint_dpi;blueprint_dpi=600
    global pca_dimension;pca_dimension=64
    
    
    ######## build model
    model=bulid_model()
    from tensorflow.keras.optimizers import RMSprop
    # RMSprop(learning_rate=1e-4)
    model.compile(optimizer=RMSprop(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    # plot model
    from tensorflow.keras.utils import plot_model
    if print_model_blueprint:
        print("generate image...")
        plot_model(model,show_shapes=True,to_file="model005.png",dpi=blueprint_dpi)
        print("generate image complete")
    
    
    
    
    ######## train model
    
    # load dataset
    x_train, y_train, x_test, y_test, list_of_cata = load_dataset(r"dataset_new\iPSC_Morphologies")
    print(list_of_cata)
    print(x_train.shape)
    print(y_train.shape)

    
    if train_new_model:
        print("start training")
        train_history = model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=epochs,
            verbose=1,
            validation_split=0.2,
        )
        model.save(rf"mast_test\model\model005_{input_seed}.keras")
        plot_train_history(train_history)
    else:
        model=load_model(rf"mast_test\model\model005_{input_seed}.keras")
        print("load trained model")



    ######## valuate model

    from sklearn.metrics import classification_report
    loss, accuracy = model.evaluate(x_test, y_test)
    
    print("test loss: ", loss)
    print("test accuracy: ", accuracy)
    from sklearn.metrics import classification_report
    report_fully_connect=classification_report(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
    print(report_fully_connect)
    # print(model.layers)
    
    
    ######## get cnn model
    for i in model.layers:
        if type(i)==type(layers.Flatten()):
            CNN_model=Model(inputs=model.input,outputs=i.output)
            CNN_model.summary()
            x_train_features = CNN_model.predict(x_train)
            x_test_features = CNN_model.predict(x_test)
            y_train_cata=np.argmax(y_train, axis=1)
            y_test_cata=np.argmax(y_test, axis=1)
            break
    print("get cnn model")
    
    print(x_train_features.shape)
    print(x_train_features[0][:10])
    
    
    scaler=StandardScaler()
    x_train_features=scaler.fit_transform(x_train_features)
    x_test_features=scaler.transform(x_test_features)
    print(x_train_features.shape)
    print(x_train_features[0][:10])
    
    ######## plot 
    from umap import UMAP
    umap = UMAP(n_components=3, random_state=42)
    x_train_embedded = umap.fit_transform(x_train_features)
    x_test_embedded = umap.transform(x_test_features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train_embedded[:,0],x_train_embedded[:,1],x_train_embedded[:,2],c=y_train_cata,cmap='coolwarm',s=5)
    ax.scatter(x_test_embedded[:,0],x_test_embedded[:,1],x_test_embedded[:,2],c=y_test_cata,cmap='coolwarm',s=30)
    
    
    
    
    
    
    ######## PCA
    from sklearn.decomposition import KernelPCA,PCA
    print("start training PCA ...")
    # pca = PCA(n_components=0.85)
    del x_train, x_test
    pca = KernelPCA(kernel='rbf', n_components=pca_dimension,gamma=1e-5)
    x_train_pca = pca.fit_transform(x_train_features)
    x_test_pca = pca.transform(x_test_features)
    
    print("pca demention:" ,x_train_pca.shape[-1])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train_pca[:,0],x_train_pca[:,1],x_train_pca[:,2],c=y_train_cata,cmap='coolwarm',s=5)
    ax.scatter(x_test_pca[:,0],x_test_pca[:,1],x_test_pca[:,2],c=y_test_cata,cmap='coolwarm',s=30)

    print("successfully trained PCA")
        
    # plt.show()
    
    print(x_train_pca.shape)
    print(x_train_pca[0][:100])
    
    
    
    ######## LDA
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics.pairwise import rbf_kernel

    lda = LinearDiscriminantAnalysis(n_components=4)
    x_train_lda = lda.fit_transform(x_train_features, y_train_cata)
    x_test_lda = lda.transform(x_test_features)
    print(x_train_lda.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train_lda[:,0],x_train_lda[:,1],x_train_lda[:,2],c=y_train_cata,cmap='coolwarm',s=5)
    ax.scatter(x_test_lda[:,0],x_test_lda[:,1],x_test_lda[:,2],c=y_test_cata,cmap='coolwarm',s=30)

    print("successfully trained LDA")
    # plt.show()
    
    ######### train pca+svm
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.base import clone


    # svm = LinearSVC(C=1, max_iter=int(1e5), verbose=1)
    # svm = LinearSVC(C=1, max_iter=int(1e5))
    pca_svm = SVC(kernel='poly', degree=3, C=100, coef0=1, verbose=1,decision_function_shape='ovo')
    
    print("start training svm...")
    pca_svm.fit(x_train_pca, y_train_cata)
    print("training end")
    y_predict=pca_svm.predict(x_test_pca)
    report_pca_svm = classification_report(y_test_cata, y_predict)
    
    
    
    ######### train svm
    
    svm = clone(pca_svm)
    print("start training svm...")
    svm.fit(x_train_features, y_train_cata)
    print("training end")
    y_predict=svm.predict(x_test_features)
    
    report_svm = classification_report(y_test_cata, y_predict)
    
    
    ######### train LDA+svm
    
    lda_svm = clone(pca_svm)
    print("start training svm...")
    lda_svm.fit(x_train_lda, y_train_cata)
    print("training end")
    y_predict=lda_svm.predict(x_test_lda)
    
    report_lda_svm = classification_report(y_test_cata, y_predict)
    
    
    
    
    print("==============fully connected:==============\n",report_fully_connect)
    print("==============svm w/o pca:==============\n",report_svm)
    print("==============svm with pca:==============\n",report_pca_svm)
    print("==============svm with lda:==============\n",report_lda_svm)
    
    # plt.show()
    print("program end")
    return [input_seed,report_fully_connect,report_svm,report_pca_svm,report_lda_svm,train_history.history]


# reports=[0]
for i in tqdm(range(66,101,1)):
    with open(fr"mast_test\log\{i}.txt", 'w') as file_:
        sys.stdout = file_
        report=main(i)
        sys.stdout = sys.__stdout__
    joblib.dump(report,rf"mast_test\reports\report_{i}.joblib")
    
    del report
    import gc
    gc.collect()
    print(f"save report_{i}.joblib")