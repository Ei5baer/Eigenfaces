import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread

path_name = "data\lfw"



folders = sorted(os.listdir(path_name))

face_train_dict = {}
face_test_dict = {}


#Sorting faces into test and train dictionaries and the keys of those dictionries are names of the people i.e. folders

folders = sorted(os.listdir(path_name))
for folder in folders:
    files = [os.path.join(path_name , folder, path) for path in os.listdir(os.path.join(path_name , folder))]
    if len(files) == 1:
        face_test_dict[folder] = files
    else:
        face_train_dict[folder] = files[:int(len(files)/2)]
        face_test_dict[folder] = files[int(len(files)/2):]

wh = 100 #width and height are defined as 100, so image will be 100 X 100

for face in face_train_dict: #In this module, we resize and equalize the image histogram to make the images workable, also convert them to greyscale
    for i in range(len(face_train_dict[face])):
        img_file =  face_train_dict[face][i]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wh, wh), interpolation = cv2.INTER_AREA)
        img = cv2.equalizeHist(img)
        face_train_dict[face][i] = img
        #cv2.imshow('image', img)

        
face_vec_list = []

#We generate the covariance matrix in this module

for face in face_train_dict:
    for i in range(len(face_train_dict[face])):
        face_eq = face_train_dict[face][i] 
        face_eq = np.array(face_eq) 
        #print (np.shape(face_eq))  
        face_vec = face_eq.flatten()
        #print (face_vec)
        face_vec_list.append(face_vec)
fva = np.array(face_vec_list)
orig_fva = np.array( fva )
s = np.shape( fva)
#print (s)
lens  = 0

#Average face is generated

average_face_vector = np.zeros(10000)
for i in face_train_dict:
    for j in face_train_dict[i]:
        s = j.flatten()
        lens = lens + 1
        average_face_vector = np.add(average_face_vector,s)
average_face_vector = np.divide(average_face_vector,lens).flatten()
#print("Size of average face vector:",average_face_vector.shape)
#print("Average face vector:",average_face_vector)
#print("Average face:")
#plt.imshow(average_face_vector.reshape(100, 100), cmap='gray')#plotting the average face generated
#plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')#removing the labels from the plot
#plt.show()


#Eigenfaces are calculated with th help of the LinAlg library provided by numpy

mean_fva = fva.mean(axis = 0)

fva = fva - mean_fva
fvat = fva.T
#t = np.shape(fvat)
#print(t)
Cov_mat = np.dot(fva, fvat)
a = np.shape(Cov_mat)
#print (a)
print ("done")

eigval, eigvec=np.linalg.eig(Cov_mat)

#print ("eigvals are " , eigval)
list_eigvals = []

for x in np.nditer(eigval):
    x_abs = abs(x)
    list_eigvals.append(x_abs)

eigvals = np.array(list_eigvals)

ind  = np.argsort(eigvals)
sort_eigvals = np.sort(eigvals)
eigveclist = eigvec.tolist()
#print (eigvec)

a = ind[-1]
eigfac = np.array(eigveclist[a])


eigenfaces = []


#Choosing the vectors with highest variation (depending on the highest absolute eigenvalue)

for x in range (-2,-21,-1):
    a=ind[x]
    eigfac = np.vstack((eigfac,np.array(eigveclist[a])))

eigfac_correct = eigfac.T

eigfac_correct = np.array(eigfac_correct)
eigenfac = np.dot(fvat,eigfac_correct)

eig = eigenfac.T
eige = list(eig)

for i in range (0, 20):
    eig_fac = eige[i]
    eig_face = eig_fac.reshape(100, 100)
    #cv2.imshow('eigface', eig_face)
    #cv2.imwrite("eigenface" + str(i) + ".jpg", eig_face)
    print ("i")

    #cv2.waitKey()

#print (np.shape(eigenfac))

#Working on the test set
#Editing the test faces just as we edited training faces
for face in face_test_dict:
    for i in range(len(face_test_dict[face])):
        img_file =  face_test_dict[face][i]
        img = cv2.imread(img_file)  
        #img = img.resize((100, 100), Image.ANTIALIAS)
        #img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wh, wh), interpolation = cv2.INTER_AREA)
        img = cv2.equalizeHist(img)
        face_test_dict[face][i] = img
        #cv2.imshow('image', img)

weights_test = {}

tpr = []
fpr = []

#Computing weights by matrix multiplication of test face vector with eigenfaces matrix

for face in face_test_dict:
    for i in range(len(face_test_dict[face])):
        face_eq = face_test_dict[face][i] 
        face_eq = np.array(face_eq) 
        #print (np.shape(face_eq))  
        face_vec = face_eq.flatten()
        #print (face_vec)
        face_vec_list.append(face_vec)
        face_test_dict[face][i] = face_vec
        weights = np.dot(face_vec - mean_fva , eigenfac)
        if face not in weights_test:
            j = []
        j.append(weights)
        weights_test[face] = j

#Computing distances

import math as mat
def dist(x, y):
    t = x - y
    diff_list = list(t)
    diff_sq = list(map(lambda k: k**2 , diff_list)) 
    dist_s = sum(diff_sq)
    dist_s = mat.sqrt(dist_s)
    return dist_s

#Finding out tpr and fpr and accuracy by iterating over different threshold values

for threshold in range (150, 400):
    #threshold = 270
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in weights_test:
        for x in range(len(weights_test[i])):
            for j in weights_test:
                for y in range(x, len(weights_test[j])):
                    if (np.array(weights_test[i][x]).any != np.array(weights_test[j][y]).any):
                        diste = dist(weights_test[i][x], weights_test[j][y])
                        #print(dist)
                        if(diste < threshold):
                            if (i == j):
                                #print("Correctly Verified")
                                #print ("tp")
                                tp += 1
                            else:
                                fp+=1
                        else:
                            if (i != j ):
                                #print("Correctly Unverified")
                                tn += 1
                            else:
                                fn+=1
                            
    #print ("Works")

    # Calculating accuracy
    accuracy = ((tp + tn)/float(tp+tn+fp+fn))*100.0
    print("Accuracy : ", accuracy)
    trpr = tp/float(tp + fn)
    flpr = fp/float(fp + tn)
    tpr.append(trpr)
    fpr.append(flpr)
print ("works")
"""

#threshold = 270
tp, tn, fp, fn = 0, 0, 0, 0
for i in weights_test:
    for x in range(len(weights_test[i])):
        for j in weights_test:
            for y in range(x, len(weights_test[j])):
                if (np.array(weights_test[i][x]).any != np.array(weights_test[j][y]).any):
                    diste = dist(weights_test[i][x], weights_test[j][y])
                    #print(dist)
                    if(diste < threshold):
                        if (i == j):
                            #print("Correctly Verified")
                            #print ("tp")
                            tp += 1
                        else:
                            fp+=1
                    else:
                        if (i != j ):
                            #print("Correctly Unverified")
                            tn += 1
                        else:
                            fn+=1
                            
print ("Works")

print("True Positive : ", tp)
print("True Negative : ", tn)
print("False Positive : ", fp)
print("False Negative : ", fn)

# Calculating accuracy
accuracy = ((tp + tn)/float(tp+tn+fp+fn))*100.0
print("Accuracy : ", accuracy)

"""
"""

fpr = [0.002,0.004,0.006,0.011,0.018,0.030,0.040,0.061,0.077,0.102,0.138,0.163,0.199,0.248,0.279,0.333,0.350,0.366,0.391,0.431,0.510,0.548,0.551,0.731,0.853]
tpr = [0.014,0.022,0.029,0.043,0.063,0.096,0.112,0.161,0.181,0.221,0.290,0.312,0.360,0.441,0.458,0.539,0.557,0.553,0.599,0.639,0.712,0.743,0.743,0.875,0.946]
#print (len(fpr))
"""

#Plotting the ROC curve

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.autoscale(enable=True, axis='both', tight='None')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC')
plt.plot(fpr, tpr, color='green', lw=2)
plt.show()

