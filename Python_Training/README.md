This is the code we used to train and create our model in Pytorch.

We created our own model based off the paper here: 
https://ieeexplore.ieee.org/document/9531841

We trained the model using this notebook as reference:
https://www.kaggle.com/code/mosheyerachmiel/ecg-classification

The cpp file was used to export the bin files of the model from the pth file. The basic resnet_bin.py file was provided by Ashwin Bhat, some minor modifications were made for this project. The resnet_bin.py file uses Model.py and the model_fold_6.pth to generate the .bin files.
