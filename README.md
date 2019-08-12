# policeCar
A computer vision based system for detecting police cars
# Preface
The aim of this tutorial is to give the reader an overview of the different steps that were taken to accomplish this project. The structure of this document is as follows.

# Pin input and ouput
You need to install Jetson.GPIO (sudo pip install Jetson.GPIO). If you try to import Jetson.GPIO you will recevie an permission error. In order to mitigate the error one has to cd to the "/sys/class/gpio" and change the permission of the two files "export" and "unexport" from only write to both read and write for all users (use sudo chmod 666 export and the same for unexport).
