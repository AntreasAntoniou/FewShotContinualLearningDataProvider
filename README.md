# Data Provider for Benchmarks for Continual Few-Shot Learning in Pytorch
The original code for the data providers and the datasets of the paper ["Benchmarks for Continual Few-Shot Learning"]().

## Introduction

Welcome. This repository includes code for the data providers that can generate samples for the task types found in ["Benchmarks for Continual Few-Shot Learning"]().
Furthermore, it provides links to both Omniglot and SlimageNet datasets.

## Installation

The code uses Pytorch to run, along with many other smaller packages. To take care of everything at once, we recommend 
using the conda package management library. More specifically, 
[miniconda3](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), as it is lightweight and fast to install.
If you have an existing miniconda3 installation please start at step 3. 
If you want to  install both conda and the required packages, please run:
 1. ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
 2. Go through the installation.
 3. Activate conda using ```conda activate```
 4. conda create -n pytorch_env python=3.6.
 5. conda activate pytorch_env
 6. At this stage you need to choose which version of pytorch you need by visiting [here](https://pytorch.org/get-started/locally/)
 7. Choose and install the pytorch variant of your choice using the conda commands.
 8. Then run ```bash install.sh```

To execute an installation script simply run:
```bash <installation_file_name>```


## Datasets
We provide functionality for both SlimageNet and Omniglot. We have automated the unzipping and usage of the datasets, all one needs to do is download them from:

- [SlimageNet repository]()
- [Omniglot part_1](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip)
- [Omniglot part_2](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip)

Once downloaded, please place them in the datasets folder in this repo. The rest will be done automagically when you 
run an experiment.

For **Omniglot**, unzip the two folders and mix their contents into a single folder which should then be placed under the 
datasets folder.

**Note**: By downloading and using the SlimageNet dataset, you accept terms and conditions found in [imagenet_license.md](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/imagenet_license.md) 

#### Other Datasets:
We provide a mechanism for quick and easy training of models on any image-based datasets. 
Read the data.py description in the [Code Overview](#code-overview) Section for details on how to train models on your own datasets.  

## Code Overview:

- datasets folder: Contains the dataset pbzip files and folders containing the images in a structure readable by the 
custom data provider.
- utils: Contains utilities for dataset extraction, parser argument extraction and storage of statistics and others.
- data.py: Contains the data providers for the continual few shot meta learning task generation. The data provider is agnostic
to dataset, which means it can be used with any dataset. Most importantly, it can only scan and use datasets when they 
are presented in a specific format. The two formats that the data provider can read are:
1. A folder structure where the top level folders are the classes and the contained images of each folders, the images 
of that class, as illustrated below:
```
Dataset
    ||______
    |       |
 class_0 class_1 ... class_N
    |       |___________________
    |                           |
samples for class_0    samples for class_1
```
In this case the data provider will split the data into 3 sets, train, val and test using the train_val_test_split 
variable found in the experiment_config files. However, in the case where you have a pre-split dataset, such 
as mini_imagenet, you can instead use:
2. A folder structure where the higher level folders indicate the set (i.e. train, val, test), the mid level folders 
(i.e. the folders within a particular set) indicate the class and the images within each class indicate the images of 
that class.
```
Dataset
    ||
 ___||_________
|       |     |
Train   Val  Test
|_________________________
    |       |            |
 class_0 class_1 ... class_N
    |       |___________________
    |                           |
samples for class_0    samples for class_1
```

# Running an example usage script:

To run an experiment from the paper on Omniglot:
1. Activate your conda environment ```conda activate pytorch_env```
2. cd experiment_scripts
3. Find which experiment you want to run.
4. ```python example_usage_omniglot.py```


# Acnknowledgments
Thanks to the University of Edinburgh and ESPRC research council for funding this research.
 
 
