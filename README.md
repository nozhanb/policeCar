# policeCar
A computer vision based system for detecting police cars
# Preface
The aim of this tutorial is to give the reader an overview of the different steps that were taken to accomplish this project. The structure of this document is as follows.

# Pin input and ouput
You need to install Jetson.GPIO (sudo pip install Jetson.GPIO). If you try to import Jetson.GPIO you will recevie an permission error. In order to mitigate the error one has to cd to the "/sys/class/gpio" and change the permission of the two files "export" and "unexport" from only write to both read and write for all users (use sudo chmod 666 export and the same for unexport). Once you print out the result of the GPIO.getmode(), you will probably get 10, 11 or other digits. 10 here means GPIO.BOARD and 11 means GPIO.BCM. Check [this](https://stackoverflow.com/questions/31687465/gpio-getmode-in-python-on-raspberry-pi-gets-different-value-than-on-wiki/31688886#31688886) link.

After installing the Jetson.GPIO library follow the instructions on [this](https://github.com/NVIDIA/jetson-gpio) page (under the Setting User Permissions section). You need to create a new group, add your username to the group and cp a ".rules" file to the given path in the /etc/... (see the instructions). Finally, you need to restart the Jetson so the permissions take effect. If you do not follow the instruction one has to change all the files permissions manually and some cases the termial freezes. You save your self alot of time by following the instructions on permissions.
