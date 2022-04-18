It is done, with extreme simplicity, an image recognition process contained in the LFW database http://vivw.cs.umass.edu/LFW/#DOWNLOAD, taking advantage of the Sklearn ease to implement the SVM model. The recognition of cascaded faces is also used to debug the images by obtaining precisions exceeding 70% in the test with images that are not listed in the training.

Requirements:

Have Python installed (not necessarily the latest version, even with which the tests have been carried out) and installed the packets corresponding to

Import OS.

Import Re

Import CV2.

Import Numpy AS NP

from Sklearn.svm Import SVC

Import Pickle #to Save the Model

from Sklearn.multiclass Import Onevsrestclassifier

It is advisable to have an ACONDA installed and work with Spyder from Anaconda, which guarantees an integrated and friendly environment, installing any package that is missing from the CMD.EXE Prompt of Anaconda with type commands:

Python -M Pip Install OpenCV-Python (CASE OF CV2)

You must have the image recognition module at the Haarcascade_frontalface_default.xml, in this case it has been downloaded from the page https://github.com/shantnu/facedetect/ with the corresponding license recognitions. Although it can also be downloaded from other pages as in links from the recommended https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/

Functioning:

The test files are accompanied:

LFW3.ZIP containing betting images downloaded from http://vis-ww.cs.umass.edu/lfw/#download and specifically from the option All Images Aligned With Commercial Face Alignment Software (LFW-A - Taigman, Wolf , Hassner)

LFW3_Test.zip containing the used test images. The images listed in the test have been eliminated from training since in another case there would be a very high acceleration rate that would not correspond to reality.

Both files, as well as the haarcascade_frontalface_default.xml module should be downloaded (and unzipped the .zip) on the D disc:, in another case you have to change the DIRNAME and DIRNAME_TEST parameters at the beginning of the GUESSIMAGESSVM_FACES.PY program

The setting of the FaceCascade ScaleFactor = 2.1 parameter is highlighted, from the value 1.1 with which it is usually included, which produces an automatic debugging of images that allows obtaining success rates over 70%. An extensive and affordable explanation about the parameters of FaceCascade is included in the recommended article https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/

The program produces a list of images that have not passed the purification

The disposition of the images of the LFW database, in which the name of person appears in the name of the image has allowed its labeling per program.

The program to execute is ::

Guessimagessvm_face.py.

References:

http://vis-ww.cs.umass.edu/lfw/#download.
https://github.com/shantnu/facedetect/
https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/
https://realpython.com/face-recognition-with-python/
