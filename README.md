# the aim of the project is to build a classifer that can identify differenc iPSc

## model description 
1. model001:cnn+fully_connected_layer
2. model002:cnn+SVM
3. model003:cnn+SVM+standardized
4. model004:cnn+SVM+multisensor
5. model005:cnn+PCA+SVM+multisensor total 4 model included
6. model005:cnn+PCA+SVM+multisensor total 4 model included with data argumentation

## for more information
see **final_presentation.pptx** and **final_report.pdf**

## reference:


1.	Download Cell Data: Images, Genomics, & Features. (2018). ALLEN CELL EXPLORER. https://www.allencell.org/data-downloading.html#DownloadFeatureData
2.	Lien, C.-Y., Chen, T.-T., Tsai, E.-T., Hsiao, Y., Lee, N., Gao, C.-E., Yang, Y., Chen, S., Yarmishyn, A. A., Hwang, D., Chou, S., Chu, W.-C., Chiou, S., & Chien, Y. (2023). Recognizing the Differentiation Degree of Human Induced Pluripotent Stem Cell-Derived Retinal Pigment Epithelium Cells Using Machine Learning and Deep Learning-Based Approaches. Cells, 12(2), 211–211. https://doi.org/10.3390/cells12020211
3.	Pfaendler, R. (2022). Morphologically annotated single-cell images of human induced pluripotent stem cells for deep learning. Ethz.ch. https://doi.org/10.3929/ethz-b-000581447


## folder structure

``` bash
├─..bfg-report # bfg
│  └─...
│ 
├─.git 
│  └─...
│
├─.history # all code code copy by vscode extenstion Local History
│  └─...
│
├─.trash # trash backup
│  └─...
│
├─.vscode # vscode setting
│  └─...
│
├─assest_csv # quilt dataset csv file not used
│  └─...
├─assest_image # quilt dataset image file not used
│  └─...
├─dataset_new # dataset used
│  ├─iPSC_Morphologies
│  │  ├─test
│  │  │  ├─Big
│  │  │  ├─Long
│  │  │  ├─Mitotic
│  │  │  ├─RAR-treated
│  │  │  └─Round
│  │  └─train
│  │      ├─Big
│  │      ├─Long
│  │      ├─Mitotic
│  │      ├─RAR-treated
│  │      └─Round
│  └─iPSC_QCData
│      ├─test
│      │  ├─Cell
│      │  ├─Debris
│      │  ├─DyingCell
│      │  └─MitoticCell
│      └─train
│          ├─Cell
│          ├─Debris
│          ├─DyingCell
│          └─MitoticCell
│
├─data_in_model006 # result of model006
│  └─...
│
├─data_in_model007 # result of model005
│  └─...
│
├─mast_test # test with many seed to identify the accuracy
│  └─...
│
├─ref # reference paper
│  └─...
│
├─visualize # visualize the data
│  └─...
│
├─__pycache__ # cache
│  └─...
├─01_read_data.py # read data in dataset
├─02_generate_test.py # split train and test 
├─03_train_model001.py # cnn+fully_connected_layer
├─04_train_model002.py # cnn+SVM
├─05_train_model003.py # cnn+SVM+standardized
├─06_train_model004.py # cnn+SVM+multisensor
├─07_train_model005.py # cnn+PCA+SVM+multisensor total 4 model included
├─08_train_model006.py # cnn+PCA+SVM+multisensor total 4 model included with data argumentation
├─bfg.jar
├─calculate_dataset_Statistics.csv # the max and min value in total 7 channel
├─calculate_dataset_Statistics.py # calculate the max and min value in total 7 channel
├─data_in_model00sx_test_pca.joblib
├─final_presentation.pptx
├─final_report.pdf # final report
├─model001.keras # model of 001
├─model002.keras # model of 002
├─model002.png
├─model003.png
├─model004.keras  # model of 004
├─model004.png # blueprint of 004
├─model005.keras # model of 005
├─model005.png# blueprint of 005
├─model006.keras # model of 006
├─model006.png # blueprint of 006
├─mylib.py
├─README.md
├─requirements.txt # environment requirements
├─train_history copy.svg
└─train_history.svg
```