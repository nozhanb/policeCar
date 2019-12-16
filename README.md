# PoliceCar
A computer vision model for (Norwegian) police car detection.

# Installation
In order to run the police car detection model the following parts are necessary 
1. Darkflow
2. Jetson nano developer kit

# 1. Darkflow
The Darkflow model is the implementation of [YOLO](https://pjreddie.com/darknet/yolo/) (originally written in C) in Python/TensorFlow. For a detailed overview of Darkflow the reader is encouraged to check the Darkflow [github](https://github.com/thtrieu/darkflow) repository where the installtion process along with other useful information can be found. For a short instruction on Darkflow installtion please read section 1.1. For the rest of this tutorial it is assumed that the user is working on a linux system.

## 1-1- Darkflow Installation
According to Darkflow [page](https://github.com/thtrieu/darkflow) there are three different ways to install Darkflow. In this tutorial we will follow the first method. First, you need to navigate to the Darkflow [page](https://github.com/thtrieu/darkflow). Next, click on the "clone or download" to download the entire repository as ".ZIP" file or use the url to clone the repository on your computer (you need to have ___git___ insalled on your computer). If zip method is used then unzip the contents of the file. After unziping the contents navigate to the "root" directory where the ___setup.py___ along with other files and directories(e.g. cfg directory) is located. On the command line (inside the root directory where the setup.py file is located) enter the following command (make sure you have python3 installed on your system):

>python3 setup.py build_ext --inplace

Depending on whether you have the necessary python packages on your system you may receive error messages! ___Cython___ and ___opencv-python___ are the two most likely packages that you may get error messages about. Make sure you have them installed and python has access to them. Once the setup.py has run successfully, you can test the Darkflow model by exceuting the following command in the root directory.

>flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0

Flow is the executive command. By executing the above line the Darkflow looks inside the ___sample_image___ directory for test images. Next, it uses ___tiny-yolo.cfg___ configuration file located inside ___cfg___ directory. Then, it takes the ___tiny-yolo.weights___ located inside ___bin___ directory. If your system has GPU installed on it and enabled it will run on gpu. If the model ran successfully you would be able to see the results inside ___out___ directory located in ___sample_image___.

## 1.2. Darkflow Configuration File
You can have access to several different configuration files inside ___cfg___ directory. There are several cfg files in the cfg directory each for a different YOLO flavor (including tiny-YOLO, YOLO-VOC, and YOLO; read this [page](https://pjreddie.com/darknet/yolo/) for a description of each flavor). Each cfg file contains the structure of the model along with the parameters for traianing that model (e.g. learning rate, decay rate, momentum). For training a model the cfg file and the path to the cfg file's location has to be given along with the --model flag (similar to the code above).

## 1.3. Darkflow Weights
Depending on the model you are planning to use the appropriate weight file needs to be downloaded from [this](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) google drive. (The same link can be accessed by navigating to the ___intro___ section of [this](https://pjreddie.com/darknet/yolo/) page.) Once the right weight file has downloaded put it inside the ___bin___ directory so the flow can access to it (e.g. bin/tiny-yolo.weights).

## 1.4. Darkflow Annotations, Classes
To create your annotation file you need to follow the YOLO annotation format which requires a ___.xml___ format. For the purpose of the car detection model [labelImg](https://github.com/tzutalin/labelImg) was used. LabelImg will allow you to store annotations and class names in both ___.txt___ and ___.xml___ formats.

[comment]: <> (# 1.5. Training Darkflow on a Custom Dataset)


# 2. Darkflow Car Detection
In order to train and run Darkflow on your own custom dataset, you need to have the required files and directories [Correct this part ]

- ___annotations___ directory
- ___bin___ directory
- ___built_graph___ directory
- ___cfg___ directory
- ___ckpt___ directory
- ___images___ directory
- ___labels.txt___ file 

## 2.1 Darkflow Car Detection File System
The following shows the structure of the Darkflow police car detection model:

```bash
root
|
|___ annotations
|              |___ annotation.xml (your .xml files)
|
|___ bin
|       |___ weight files (the original and renamed weight files)
|
|___ built_graph
|              |___ .meta and .pb files
|
|___ cfg
|      |___ .cfg files (the original and renamed one)
|
|___ ckpt
|       |___ check point files will be stroed here
|
|___ images
|         |___ your .png or .jpeg (here police car images)
|
|___ labels.txt (here only police car label/class)

```
### 2.1.1 Annotaitons Directory
Where you keep the annotaions files. In case of this model there are as many .xml files as there are images of police cars. Each .xml file gives the coordinates of the ground truth box. For instance, example_01.xml files contains the coordiantes of the ground truth box around a police car in the image 01.

### 2.1.2 Bin Directory
Where you keep your weight files. Notice that the Darkflow requires you to provide it with two identical weight files with different file names. For examples if you are using tiny-yolo model, you have to keep the original name of the weight file that is ___yolov2-tiny-voc.weights___ while changing the name of the second weight file to something different (e.g. ___yolov2-tiny-c2.weights___). Please read the 5th part of the ___Training on your own dataset___ section [here](https://github.com/thtrieu/darkflow) for more details. 

### 2.2.3 CKPT Directory
This is the ___checkpoint___ directory. During the traingin session Darkflow will store a group of four different files inside this directory. This process happens repeatedly after a given number of epochs. An example is given below;

1- ___yolov2-tiny-voc_c2-250.data-00000-of-00001___ 

2- ___yolov2-tiny-voc_c2-250.index___

3- ___yolov2-tiny-voc_c2-250.meta___ 

4- ___yolov2-tiny-voc_c2-250.profile___

For the rest of this tutorial we need two of these files. First, the file with ___.data___ which contains the updated weights after 250 epochs. And the ___.meta___ file. See the ___Training new model___ section [here](https://github.com/thtrieu/darkflow) for further details on how to test your model from a given checkpoint. 

### 2.2.4 Built_graph Directory
Once the desired checkpoint has achived the following command has to be applied to convert the ___.data___ file to a ___protobuf file (.pb)___. The .pb file is the weight file that will be used in Jetson nano developer kit.

>flow --model cfg/yolo-new.cfg --load -1 --savepb

>flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb

Remember that the ___.weight___ file in the command line above has to be replaced by desired ___.data___ file from the checkpoint (ckpt) directory. Please read the ___Save the built graph to a protobuf file (.pb)___ [here](https://github.com/thtrieu/darkflow) for more details.

### 2.2.5 cfg Directory
Put your configuration file inside this directory (e.g. yolov2-tiny-voc.cfg). There are two lines that need to be changed when it comes to training on custom datasets; number of ___classes___ and number of ___filters___. Before making any changes to the any files remember that, similar to the weight files, a copy of the desired configuration file with a different name needs to be made and placed in the cfg directory. For instance, one may decide to use the original ___yolov2-tiny-voc.cfg___ file. Then, one has to make another copy of the same file with a different name, e.g. ___yolov2-tiny-voc_c2.cfg___, and place it along with the original file in the cfg directory. Next, comes the filter and class changes. These changes need to be done to the ___renamed___ configuration file. At the end of the renamed file you can find both classes and filters parameters. Place the number of classes you are going to train your model on (in case of police car the original value is replaced with 1). However, for the numner of filters you have to follow the following equation;

> num * (classes + 5)

Where one has to replace ___num___ with 5 and classes with the number classes you are training on. For example, in case of Police car detection model the number of classes is 1 and num is 5 which according to the equation the number of filters becomes 5*6 = 30. Thus,  the original value of the filter in the renamed cfg file will be replaced with 30 (from 125 to 30). The user has to use the modified and renamed configuration file when training their model!  Please read ___Training on your own dataset___ section [here](https://github.com/thtrieu/darkflow) for more details.

### 2.2.6 Images Directory
This directory contains your training images (i.e. police car images).

### 2.2.7 labels.txt file
This file contains the label/class of the objects to be detected (i.e. police car)

## 2.3 Training and Inference
Once you have built the required file system and place the necessary files inside the correct directory you can run the following command to start traingin your model. 

>flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights --train --gpu 1.0

The above line uses yolo-new configuration file along with tiny-yolo weights and runs on a GPU. For a complete list of manditory and optional flags type in the following command on the command line: ???????? [add the command]



# 3. Jetson Nano Developer Kit

## 3.1 Installation
You can find a good introduction on seting up your Jetson nano develper kit [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro), however, below you can find a very short summary of the steps you need to take to prepare your Jetson device.

1- You need to have a microSD card

2- Download ___Jetson Nano Developer Kit SD Card Image___ [here](https://developer.nvidia.com/jetson-nano-sd-card-image-r3223)

3- Write the image to your microSD card using [Etcher](https://www.balena.io/etcher/)(pick the one that suits your operating system)

4- After writing the image to your microSD card insert the card into your Jetson device

5- Power on and you are good to go!


## 3.2 Running Your Model on Jetson

The operating system on the Jetson is a linux operting system so to install the necessary python packages you can simply use the  ___pip___ command. Once you have the python packages installed you need to install the Darkflow on Jetson following one of the three methods provided [here](https://github.com/thtrieu/darkflow#getting-started) (similar to [section]() ) on Jetson you need to create a similar file system structure (see section ___2.1___). However, this time you only need to provide a few directories including; ___built_graph___, 

[create an anchor](#1-1-darkflow-installation)


# 3.1 Pin input and ouput
You need to install Jetson.GPIO (sudo pip install Jetson.GPIO). If you try to import Jetson.GPIO you will recevie a permission error. In order to mitigate the error one has to cd to the "/sys/class/gpio" and change the permission of the two files "export" and "unexport" from only write to both read and write for all users (use sudo chmod 666 export and the same for unexport). Once you print out the result of the GPIO.getmode(), you will probably get 10, 11 or other digits. 10 here means GPIO.BOARD and 11 means GPIO.BCM. Check [this](https://stackoverflow.com/questions/31687465/gpio-getmode-in-python-on-raspberry-pi-gets-different-value-than-on-wiki/31688886#31688886) link.

After installing the Jetson.GPIO library follow the instructions on [this](https://github.com/NVIDIA/jetson-gpio) page (under the Setting User Permissions section). You need to create a new group, add your username to the group and cp a ".rules" file to the given path in the /etc/... (see the instructions). Finally, you need to restart the Jetson so the permissions take effect. If you do not follow the instruction one has to change all the files permissions manually and some cases the termial freezes. You save your self alot of time by following the instructions on permissions. 

Initially, I used the gpio 12 on Jetson but it turned out to be the wrong pin. When I turned the videoplayer on and clicked on the play botton the led would turn on! And when clicked on the pause botton the led would turn off after a few second! And this process can be repeated as many times as you click on the play botton. I set the pin 12 as an output inside the predict.py file (/usr/local/lib/python3.6/dist-packages/darkflow/net/yolov2/predict.py)!?. I am not sure why there is a connection between pin 12 and the videoplayer!?!

Remember that there are four types of mode to be set but the two that are relavent for this work are "BOARD" and "CBM" (read [this](https://github.com/NVIDIA/jetson-gpio) link for more details). The import thing to remember is to if you set your mode to BOARD you should use pin numbering on the board (e.g. 12, 26) and when CBM is the choice gpio numbering must be used (e.g. "79" for 12).

# Useful Links
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

https://www.element14.com/community/community/designcenter/single-board-computers/blog/2019/05/21/nvidia-jetson-nano-developer-kit-pinout-and-diagrams

https://pinout.xyz/pinout/uart#



# How to install Nvidia and Cuda for GPU on Ubunut18.04

Follow the instructions [here](https://www.tensorflow.org/install/gpu). Install, tensorflow and tensorflow-gpu separately (I am installing version 1.14 for both tesnorflow and tesnsorflow-gpu). In case you run into problem during the installation of cuda;

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

Reading package lists... Done

Building dependency tree       

Reading state information... Done

You might want to run 'apt --fix-broken install' to correct these.

The following packages have unmet dependencies:

libnvinfer-dev : Depends: libcublas-dev but it is not installed

E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).

@@@@@@@@@@@@@@@@@@@@@@@@@@
and then the following happens:

sudo apt --fix-broken install
Reading package lists... Done
Building dependency tree       
Reading state information... Done
Correcting dependencies... Done
The following packages were automatically installed and are no longer required:

  libatomic1:i386 libbsd0:i386 libdrm-amdgpu1:i386 libdrm-intel1:i386 libdrm-nouveau2:i386 libdrm-radeon1:i386 libdrm2:i386 
  
  libedit2:i386 libelf1:i386 libexpat1:i386 libffi6:i386 libgl1:i386
  
  libgl1-mesa-dri:i386 libglapi-mesa:i386 libglvnd0:i386 libglx-mesa0:i386 libglx0:i386 libllvm7 libllvm7:i386 libllvm8:i386 
  
  libnvidia-common-396 libpciaccess0:i386 libqgis-3d3.8.0 libqgis-analysis3.8.0
  
  libqgis-app3.8.0 libqgis-core3.8.0 libqgis-gui3.8.0 libqgis-native3.8.0 libqgis-server3.8.0 libqgisgrass7-3.8.0 
  
  libqgispython3.8.0 libsensors4:i386 libstdc++6:i386 libx11-6:i386 libx11-xcb1:i386
  
  libxau6:i386 libxcb-dri2-0:i386 libxcb-dri3-0:i386 libxcb-glx0:i386 libxcb-present0:i386 libxcb-sync1:i386 libxcb1:i386 
  
  libxdamage1:i386 libxdmcp6:i386 libxext6:i386 libxfixes3:i386 libxshmfence1:i386
  
  libxxf86vm1:i386

Use 'sudo apt autoremove' to remove them.

The following additional packages will be installed:

libcublas-dev

The following NEW packages will be installed:

libcublas-dev

0 upgraded, 1 newly installed, 0 to remove and 82 not upgraded.

1 not fully installed or removed.

Need to get 0 B/38.9 MB of archives.

After this operation, 109 MB of additional disk space will be used.

Do you want to continue? [Y/n] y

(Reading database ... 385338 files and directories currently installed.)

Preparing to unpack .../libcublas-dev_10.2.1.243-1_amd64.deb ...

Unpacking libcublas-dev (10.2.1.243-1) ...

dpkg: error processing archive /var/cache/apt/archives/libcublas-dev_10.2.1.243-1_amd64.deb (--unpack):

trying to overwrite '/usr/lib/x86_64-linux-gnu/libcublas_static.a', which is also in package nvidia-cuda-dev 9.1.85-3ubuntu1

dpkg-deb: error: paste subprocess was killed by signal (Broken pipe)

Errors were encountered while processing:

/var/cache/apt/archives/libcublas-dev_10.2.1.243-1_amd64.deb

E: Sub-process /usr/bin/dpkg returned an error code (1)

@@@@@@@@@@@@@@@
Follow [this](https://devtalk.nvidia.com/default/topic/1048021/cuda-setup-and-installation/error-depends-libcublas-dev-gt-10-1-0-105-but-it-is-not-installed-ubuntu-18-04/) instruction;

1    sudo prime-select intel

2    sudo mv /etc/apt/sources.list.d/cuda.list $HOME/cuda.list.bak

3    sudo apt update

4   sudo apt-get purge nvidia-*

5    sudo apt-get -f install

6    sudo apt update

7    sudo apt autoremove

Then you should be fine!

