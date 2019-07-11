import math as mat
import numpy as np
#import cv2 
import matplotlib.pyplot as plt
import os
import cv2
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

testimage_filenames = {}
trainimage_filenames = {}
print("hello")
for path,dirs,files in os.walk("data\lfw"):
    print ("path is ", path )
    prev = None
    for file in files :
        print ("file name is ", file )
        if ( file.endswith('jpg') ) :
            #image_filenames.append( path+ "/" +file )
            tokens = path.split("\\")
            key = tokens[-1]
            print("key is ", key )
            tok = file.split(".")
            namennum  = tok[0]
            print ( "namennum is " , namennum )
            name_key = namennum.split("_")
            name = name_key[0] + name_key[1]
            num = name_key[-1]
            
            num = int(num)
            print (num)
            if (name != prev):
                full_listtrain = []
                full_listtest = []

            print ("num is ", num )
            if not ( name in  trainimage_filenames ) :
                trainimage_filenames[name] = []
            if not ( name in  testimage_filenames ) :
                testimage_filenames[name] = []

            if(num % 2 != 0):
                (testimage_filenames[name]).append(path + "\\" +file) 
            if (num % 2 == 0):
                (trainimage_filenames[name]).append(path + "\\" +file )
            prev = name

print( trainimage_filenames )
#exit()

#for dir in dirs:


cropped_face_dict_train = {}
cropped_face_dict_test = {}

histeq = {}



images = {}
cropped_faces = []

#def convertToRGB(image):
#    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import shutil
shutil.rmtree('./training_dataset_crop', ignore_errors=True)
shutil.rmtree('./testing_dataset_crop', ignore_errors=True)

for name in trainimage_filenames :
    path_name = "training_dataset_crop" + "//" + name
    os.makedirs(path_name)
    if not ( name in cropped_face_dict_train ) :
        cropped_face_dict_train[name] = []
    count = 0 

    for img_file in trainimage_filenames[name]:
        # Converting to grayscale as opencv expects detector takes in input gray scale images
        img = cv2.imread(img_file)
        train_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_rects = haar_cascade_face.detectMultiScale(train_image_gray, scaleFactor = 1.2, minNeighbors = 5)
#        all_faces_of_person = all_faces_of_person + faces_rects 

        # Let us print the no. of faces found
        print('Faces found in this image : ', len(faces_rects))

        #cropped_faces = []
        for (x,y,w,h) in faces_rects:
            a = x//1 + w//2 -50
            b = y//1 + h//2 - 50
            c = x//1 + w//2 + 50
            d = y//1 + h//2 + 50
            cv2.rectangle(train_image_gray, (a, b),( c, d) , (1, 1, 1), 1)
            #train_image_gray = convertToRGB(train_image_gray)
            crop_img = train_image_gray[a:c, b:d]
            face_eq = cv2.equalizeHist(crop_img)
            #cv2.imshow("cropped_face", crop_img)
            cropped_face_dict_train[name].append(face_eq)
            img_path = path_name + "//" + name + str( count ) + ".jpg"
            count = count+1
            cv2.imwrite( img_path ,face_eq)

for name in testimage_filenames :
    path_name = "testing_dataset_crop" + "//" + name
    os.makedirs(path_name)
    if not ( name in cropped_face_dict_test ) :
        cropped_face_dict_test[name] = []
    count = 0 

    for img_file in testimage_filenames[name]:
        # Converting to grayscale as opencv expects detector takes in input gray scale images
        img = cv2.imread(img_file)
        test_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)
#        all_faces_of_person = all_faces_of_person + faces_rects 

        # Let us print the no. of faces found
        print('Faces found in this image : ', len(faces_rects))

        #cropped_faces = []
        for (x,y,w,h) in faces_rects:
            a = x//1 + w//2 -50
            b = y//1 + h//2 - 50
            c = x//1 + w//2 + 50
            d = y//1 + h//2 + 50
            cv2.rectangle(test_image_gray, (a, b),( c, d) , (1, 1, 1), 1)
            #train_image_gray = convertToRGB(train_image_gray)
            crop_img = test_image_gray[a:c, b:d]
            face_eq = cv2.equalizeHist(crop_img)
            #cv2.imshow("cropped_face", crop_img)
            cropped_face_dict_test[name].append(face_eq)
            img_path = path_name + "//" + name + str( count ) + ".jpg"
            count = count+1
            cv2.imwrite( img_path ,face_eq)
        
       
    #test_image = convertToRGB(test_image)
    #cv2.imshow("person", test_image)
"""
equ = []

name = []
#print (len( cropped_faces ))
lcf = len(cropped_face_dict)
#print (lcf)

face_vec_dict = {}
face_vec_list = []
 
for i in cropped_face_dict :
    face = cropped_face_dict[i]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
   
    dimensions = face_gray.shape

    #face_eq = cv2.equalizeHist(face_gray)
    
    face_vec = face_eq.flatten()
    face_vec_dict[i] = face_vec
    name.append(i)

    face_list_vec = list(face_vec)
    
    face_vec_list.append( face_list_vec)  
    
fva = np.array(face_vec_list)
orig_fva = np.array( fva )
s = np.shape( fva)
print (s)
mean_fva = fva.mean(axis = 0)

fva = fva - mean_fva
fvat = fva.T
#t = np.shape(fvat)
#print(t)
Cov_mat = np.dot(fva, fvat)
a = np.shape(Cov_mat)
print (a)

eigval, eigvec=np.linalg.eig(Cov_mat)

print ("eigvals are " , eigval)
#input( "proceed ? press any key " ) 

list_eigvals = []

for x in np.nditer(eigval):
    x_abs = abs(x)
    list_eigvals.append(x_abs)

eigvals = np.array(list_eigvals)

ind  = np.argsort(eigvals)

sort_eigvals = np.sort(eigvals)

eigveclist = eigvec.tolist()
print (eigvec)
a = ind[-1]
eigfac = np.array(eigveclist[a])
name_s = []
k  = name[a]
name_s.append(k)

eigenfaces = []




for x in range (-2,-5,-1):
    a=ind[x]
    k = name[a]
    name_s.append(k)
    eigfac = np.vstack((eigfac,np.array(eigveclist[a])))

eigfac_correct = eigfac.T

for x in range (-5,-13,-1):
    a=ind[x]
    k = name[a]
    name_s.append(k)

#print (eigfac_correct)

eigenfac = np.dot(fvat,eigfac_correct)

for i in range( eigenfac.shape[1] )  :
    nrm =  np.linalg.norm ( eigenfac[:,i] )
    #print ("norm is ", nrm )
    eigenfac[ : ,i ] = eigenfac[:,i] /  nrm 


s = np.shape(eigenfac)
#print(s)
print ("Bark")
#work_eig = eigenfac.T

implist = []
print ("dog")

for g in range (0 , s[1]):
    f = eigenfac[:,g]
    #print ( "l2-norm of  f is ", np.linalg.norm( f ) )
    
    m = f.reshape(100,100)
    implist.append(m)
    #print ("Bear")

weights_listed =  []
distance_list = []

for f in face_vec_list :
    f_approx = np.dot ( np.array( f)-mean_fva , np.dot( eigenfac , eigenfac.T ) ) + mean_fva
    f =  np.array(f)

    xx = f.flatten()
    #xx = xx.tolist()
    yy = f_approx.flatten()
    #yy = yy.tolist()
    t = xx - yy
    #np.shape(xx)
    #np.shape(yy)
    #print ( np.shape(t))
    diff_list = list(t)
    #map (lambda x : x*2,
    diff_sq = list(map(lambda x: x**2 , diff_list)) 
    #dist = np.hypot(*(xx - yy))
    dist_s = sum(diff_sq)
    dist_s = mat.sqrt(dist_s)
    #print ( np.shape(dist))
    distance_list.append(dist_s)

    weights = np.dot(np.array(f) -mean_fva , eigenfac)

    weights_listed.append(weights)

    #print (weights)
    
   
    #cv2.imshow( "image" , (np.array(f) ).reshape(100,100) )
    cv2.waitKey()
    
    cv2.imshow( "image" , ((np.array(f_approx ))  / 255.0   ).reshape(100,100) )
    cv2.waitKey()
    #yesorno = input("proceed (y/n) ? ")
    #if ( yesorno == 'n' ) :
        #exit()

#for i in range (s[1]):
    #cv2.imshow("image", implist[i])
    #cv2.waitKey()

#print (name_s)

names_to_weights = {}
name_to_distance = {}

for i in range (0,12):
    names_to_weights[name_s[i]] = weights_listed[i]
    name_to_distance[name_s[i]] =  distance_list[i]


print (name_to_distance)



#    s = np.dot(f.T, E)
#   j = np.dot(s,E.T)

"""