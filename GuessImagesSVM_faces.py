# -*- coding: utf-8 -*-
"""


by Alfonso Blanco García , April 2022
"""

######################################################################
# PARAMETERS
######################################################################
dirname = "C:\\lfw3"
dirname_test = "C:\\lfw3_test"

# https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# https://realpython.com/face-recognition-with-python/

# downloaded from https://github.com/shantnu/FaceDetect/
cascPath = "C:\\haarcascade_frontalface_default.xml"
 



######################################################################

import os
import re

import cv2

import numpy as np

from sklearn.preprocessing import MinMaxScaler
faceCascade = cv2.CascadeClassifier(cascPath)
#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname + "\\"
    
    images = []
    directories = []
   
    prevRoot=''
    cant=0
    
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    Y=[]
    TabNumImage=[]
    TabDenoClass=[]
    TotImages=0
    for root, dirnames, filenames in os.walk(imgpath):
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant=cant+1
                filepath = os.path.join(root, filename)
                # https://stackoverflow.com/questions/51810407/convert-image-into-1d-array-in-python
                
                image = cv2.imread(filepath)
               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
               
                # case images are not (250,250), not in LFW
                #gray1=cv2.resize(gray, (250,250))
                
                # https://omes-va.com/deteccion-de-rostros-con-haar-cascades-python-opencv/
                faces=faceCascade.detectMultiScale(
               	               
               		gray,
               		scaleFactor=2.1,
               		minNeighbors=5,
               		minSize=(30, 30)
               		
                    #flags=cv2.CASCADE_SCALE_IMAGE
                       )  
               
           	   #print("Found {0} faces!".format(len(faces)))
                if len(faces)==0 :
                   print("Cannot detect faces in "+ filename)
                   continue
                if len(faces)>1 :
                  print("It is not considered, two faces go in "+ filename)
                  continue
             
                faces1=faces[0]
                x=faces1[0]
                y=faces1[1]
                w=faces1[2]
                h=faces1[3]
                p1=y
                p2=y+h
                p3=x
                p4=x+w
	        
                gray1 = gray[p1:p2, p3:p4] 
               #https://raghul-719.medium.com/basics-of-computer-vision-1-image-resizing-97fca504cd63
                
                gray1 = cv2.resize(gray1, (140, 140),
                              interpolation=cv2.INTER_NEAREST)
                
               
               
                 # https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
                scaler = MinMaxScaler()
                 # transform data
                gray1 = scaler.fit_transform(gray1)
                
                gray1 =gray1.flatten() 
                                            
                images.append(gray1)
                if NumImage < 0:
                    NumImage=0
                Y.append(NumImage)
               
               
                TabNumImage.append(filename)
                TotImages+=1
                if prevRoot !=root:
                  
                    prevRoot=root
                    directories.append(root)
                    
                    DenoClass=filenames[0]
                    DenoClass=DenoClass[0:len(DenoClass)-9]
                    
                    
                    TabDenoClass.append(DenoClass)
    print("")
    print('directories read:',len(directories))
    
    #print('Total sum of images in subdirs:',sum(dircount))
    print('Total sum of images ',str(TotImages))
    
    return images, Y, TabNumImage, TabDenoClass
 
###########################################################
# MAIN
##########################################################


X_train, Y_train, TabNumImage, TabDenoClass = loadimages (dirname )

print( "")

for i in range(len(TabDenoClass)):
    print(TabDenoClass[i]+ " is class " + str(i))
print("")

X_test, Y_test, TabNumImage_test, TabDenoClass_test = loadimages(dirname_test)




num_classes=len(TabDenoClass)
print("Number of classes = " + str(num_classes))
x_train=np.array(X_train)

y_train=np.array(Y_train)

x_test=np.array(X_test)
y_test=np.array(Y_test)

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier


#https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
model = OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, max_iter=1000)) #Creates model instance here
model.fit(x_train, y_train) #fits model with training data

pickle.dump(model, open("./model.pickle", 'wb')) #save model as a pickled file

predictions = model.predict(x_test)

TotalHits=0
TotalFailures=0

   
print("")
print("")
print("List of successes/errors:")  
print("")     
for i in range(len(x_test)):
    DenoClass=TabNumImage_test[i]
    DenoClass=DenoClass[0:len(DenoClass)-9]
    if DenoClass!=TabDenoClass[(predictions[i])]:
        TotalFailures=TotalFailures + 1
        print("ERROR " + TabNumImage_test[i]+ " is assigned class " + str(predictions[i])
              + " " + TabDenoClass[(predictions[i])] )
              
    else:
        print(TabNumImage_test[i]+ " is assigned class " + str(predictions[i])
              + " " + TabDenoClass[(predictions[i])])
        TotalHits=TotalHits+1
         
print("")
print("Total hits = " + str(TotalHits))  
print("Total failures = " + str(TotalFailures) )     
print("Accuracy = " + str(TotalHits*100/(TotalHits + TotalFailures)) + "%") 

