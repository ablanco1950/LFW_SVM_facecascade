It is done, with extreme simplicity, an image recognition process of images contained in the LFW database http://vivw.cs.umass.edu/LFW/#DOWNLOAD, taking advantage of the Sklearn ease to implement the SVM model. The recognition of cascaded faces is also used to debug the images by obtaining precisions exceeding 70% in the test with images that are not in the training.

Requirements:

Have Python installed (not necessarily the latest version, even with which the tests have been carried out) and installed the packets corresponding to

Import OS.

Import Re

Import CV2.

Import Numpy as np

from Sklearn.svm Import SVC

Import Pickle #to Save the Model

from Sklearn.multiclass Import Onevsrestclassifier

It is advisable to have  ANACONDA installed and work with Spyder from Anaconda, which guarantees an integrated and friendly environment, installing any package that is missing from the CMD.EXE Prompt of Anaconda with type commands:

python -m pip install opencv-python (case of cv2)

You must have the image recognition module  Haarcascade_frontalface_default.xml, in this work it has been downloaded from the page https://github.com/shantnu/facedetect/ . Although it can also be downloaded from other pages as in links from the recommended https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/

Functioning:

The test files are accompanied:

lfw3.zip, the training file, containing  images downloaded from http://vis-ww.cs.umass.edu/lfw/#download and specifically from the option All Images Aligned With Commercial Face Alignment Software (LFW-A - Taigman, Wolf , Hassner)

lfw3_test.zip containing the used test images. The images listed in the test have been eliminated from the training since in another case there would be a very high hit rate that would not correspond to reality.

Both files, as well as the haarcascade_frontalface_default.xml module should be downloaded (and unzipped the .zip) on the C: disc, in another case you have to change the DIRNAME and DIRNAME_TEST parameters at the beginning of the GuessImagesSVM_faces.py. program

The setting of the FaceCascade ScaleFactor = 2.1 parameter is highlighted, from the value 1.1 with which it is usually included, which produces an automatic debugging of images that allows obtaining success rates over 70%. An extensive and affordable explanation about the parameters of FaceCascade is included in the recommended article https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/

The program produces a list of images that have not passed the purification

The disposition of the images of the LFW database, in which the name of person appears in the name of the image, has allowed its labeling by program.

The program to execute is ::

GuessImagesSVM_faces.py.

References:

http://vis-ww.cs.umass.edu/lfw/#download

https://github.com/shantnu/facedetect/

https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/

https://realpython.com/face-recognition-with-python/

The main problem is that scikit-learn-svm-svc-extremely-slow https://stackoverflow.com/questions/40077432/why-is-scikit-learn-svm-svc-extremely-slow. but with the number of records that are handled in this job it is acceptable, apart from the fact that the compilation result can be saved
