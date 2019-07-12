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


train, test = [], [] 
folders = sorted(os.listdir(path_name))
for folder in folders:
    files = [os.path.join(path_name , folder, path) for path in os.listdir(os.path.join(path_name , folder))]
    if len(files) == 1:
        face_test_dict[folder] = files
        #print ("hey")
    else:
        face_train_dict[folder] = files[:int(len(files)/2)]
        face_test_dict[folder] = files[int(len(files)/2):]

wh = 100

for face in face_train_dict:
    for i in range(len(face_train_dict[face])):
        img_file =  face_train_dict[face][i]
        #print (np.shape(img_file))
        #print(img_file)
        img = cv2.imread(img_file)
        #img = img_file  
        #img = img.resize((100, 100), Image.ANTIALIAS)
        #img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (wh, wh), interpolation = cv2.INTER_AREA)
        img = cv2.equalizeHist(img)
        face_train_dict[face][i] = img
        #cv2.imshow('image', img)

        
face_vec_list = []

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

#print (s)
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




for x in range (-2,-21,-1):
    a=ind[x]
    #k = name[a]
    #name_s.append(k)
    eigfac = np.vstack((eigfac,np.array(eigveclist[a])))

eigfac_correct = eigfac.T

eigfac_correct = np.array(eigfac_correct)
eigenfac = np.dot(fvat,eigfac_correct)

eig = eigenfac.T
eige = list(eig)

for i in range (0, 20):
    eig_fac = eige[i]
    eig_face = eig_fac.reshape(100, 100)
    cv2.imshow('eigface', eig_face)
    cv2.waitKey()

print (np.shape(eigenfac))

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
print ("Works until now")

import math as mat
def dist(x, y):
    t = x - y
    diff_list = list(t)
    diff_sq = list(map(lambda k: k**2 , diff_list)) 
    dist_s = sum(diff_sq)
    dist_s = mat.sqrt(dist_s)
    return dist_s

threshold = 270
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
