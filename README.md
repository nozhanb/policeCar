# policeCar
A computer vision model for (Norwegian) police car detection.

# Installation
In order to run the police car detection model the following parts are necessary 
1. Darkflow
2. Jetson nano developer kit

# 1. Darkflow
The Darkflow model is the implementation of [YOLO](https://pjreddie.com/darknet/yolo/) (originally written in C) in Python/TensorFlow. For a detailed overview of Darkflow the reader is encouraged to check the Darkflow [github](https://github.com/thtrieu/darkflow) repository where the installtion process along with other useful information can be found. For a short instruction on Darkflow installtion please read section 1.1. For the rest of this tutorial it is assumed that the user is working on a linux system.

# 1.1. Darkflow Installation
According to Darkflow [page](https://github.com/thtrieu/darkflow) there are three different ways to install Darkflow. In this tutorial we will follow the first method. First, you need to navigate to the Darkflow [page](https://github.com/thtrieu/darkflow). Next, click on the "clone or download" to download the entire repository as ".ZIP" file or use the url to clone the repository on your computer (you need to have ___git___ insalled on your computer). If zip method is used then unzip the contents of the file. After unziping the contents navigate to the "root" directory where the ___setup.py___ along with other files and directories(e.g. cfg directory) is located. On the command line (inside the root directory where the setup.py file is located) enter the following command (make sure you have python3 installed on your system):

>python3 setup.py build_ext --inplace

Depending on whether you have the necessary python packages on your system you may receive error messages! ___Cython___ and ___opencv-python___ are the two most likely packages that you may get error messages about. Make sure you have them installed and python has access to them. Once the setup.py has run successfully, you can test the Darkflow model by exceuting the following command in the root directory.

>flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0

Flow is the executive command. By executing the above line the Darkflow looks inside the ___sample_image___ directory for test images. Next, it uses ___tiny-yolo.cfg___ configuration file located inside ___cfg___ directory. Then, it takes the ___tiny-yolo.weights___ located inside ___bin___ directory. If your system has GPU installed on it and enabled it will run on gpu. If the model ran successfully you would be able to see the results inside ___out___ directory located in ___sample_image___.

# 1.2. Darkflow Configuration File
You can have access to several different configuration files inside ___cfg___ directory. There are several cfg files in the cfg directory each for a different YOLO flavor (including tiny-YOLO, YOLO-VOC, and YOLO; read this [page](https://pjreddie.com/darknet/yolo/) for a description of each flavor). Each cfg file contains the structure of the model along with the parameters for traianing that model (e.g. learning rate, decay rate, momentum). For training a model the cfg file and the path to the cfg file's location has to be given along with the --model flag (similar to the code above).

# 1.3. Darkflow Weights
Depending on the model you are planning to use the appropriate weight file needs to be downloaded from [this](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) google drive. (The same link can be accessed by navigating to the ___intro___ section of [this](https://pjreddie.com/darknet/yolo/) page.) Once the right weight file has downloaded put it inside the ___bin___ directory so the flow can access to it (e.g. bin/tiny-yolo.weights).

# 1.4. Darkflow Annotations, Classes
To create your annotation file you need to follow the YOLO annotation format which requires a ___.xml___ format. For the purpose of the car detection model [labelImg](https://github.com/tzutalin/labelImg) was used. LabelImg will allow you to store annotations and class names in both ___.txt___ and ___.xml___ formats.

# 1.5. Training Darkflow on a Custom Dataset

[comment]: <> (This is a comment, it will not be included)
<> (This is a comment, it will not be included)

# 2. Darkflow Car Detection File System
In order to train and run Darkflow on your own custom dataset, you need to create the required file system. The followings are the required files and directories where they should be put inside the root directory [Correct this part ]
- a ___bin___ directory
- an ___image___ directory

# You need to be more specific about the car data and training!!!



# Pin input and ouput
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

